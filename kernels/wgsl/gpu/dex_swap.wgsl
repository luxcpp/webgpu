// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// DEX Swap Acceleration Kernels for WebGPU
// Implements high-performance Uniswap v4-style AMM math
// Target: >1M swaps/sec on modern GPUs
//
// Features:
// - Batch swap computation
// - Liquidity position management
// - Route optimization
// - Price impact calculation
// - Tick math operations
//
// Part of the Lux Network GPU acceleration library
// WebGPU/WGSL implementation

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// Q96 fixed point: 2^96 for sqrt price representation
const Q96_BITS: u32 = 96u;
const Q128_BITS: u32 = 128u;

// Fee denominators (1e6 = 100%)
const FEE_DENOMINATOR: u32 = 1000000u;

// Tick bounds (Uniswap v3 style)
const MIN_TICK: i32 = -887272;
const MAX_TICK: i32 = 887272;

// Sqrt price bounds (Q64.96)
const MIN_SQRT_RATIO_LO: u32 = 4295128739u;  // Lower 32 bits
const MAX_SQRT_RATIO_LO: u32 = 1461446703485210103287u;  // Simplified

// ============================================================================
// 128-bit Unsigned Integer Emulation
// WGSL lacks native u128, so we use two u64 (represented as vec2<u32> pairs)
// ============================================================================

struct U128 {
    lo: u32,    // Bits 0-31
    lo_hi: u32, // Bits 32-63
    hi_lo: u32, // Bits 64-95
    hi: u32,    // Bits 96-127
}

fn u128_zero() -> U128 {
    return U128(0u, 0u, 0u, 0u);
}

fn u128_from_u32(x: u32) -> U128 {
    return U128(x, 0u, 0u, 0u);
}

fn u128_from_u64(lo: u32, hi: u32) -> U128 {
    return U128(lo, hi, 0u, 0u);
}

fn u128_is_zero(a: U128) -> bool {
    return a.lo == 0u && a.lo_hi == 0u && a.hi_lo == 0u && a.hi == 0u;
}

fn u128_lt(a: U128, b: U128) -> bool {
    if (a.hi != b.hi) { return a.hi < b.hi; }
    if (a.hi_lo != b.hi_lo) { return a.hi_lo < b.hi_lo; }
    if (a.lo_hi != b.lo_hi) { return a.lo_hi < b.lo_hi; }
    return a.lo < b.lo;
}

fn u128_gt(a: U128, b: U128) -> bool {
    return u128_lt(b, a);
}

fn u128_add(a: U128, b: U128) -> U128 {
    var result: U128;
    var carry = 0u;

    let sum0 = a.lo + b.lo;
    result.lo = sum0;
    carry = select(0u, 1u, sum0 < a.lo);

    let sum1 = a.lo_hi + b.lo_hi + carry;
    result.lo_hi = sum1;
    carry = select(0u, 1u, sum1 < a.lo_hi || (carry == 1u && sum1 == a.lo_hi));

    let sum2 = a.hi_lo + b.hi_lo + carry;
    result.hi_lo = sum2;
    carry = select(0u, 1u, sum2 < a.hi_lo || (carry == 1u && sum2 == a.hi_lo));

    result.hi = a.hi + b.hi + carry;

    return result;
}

fn u128_sub(a: U128, b: U128) -> U128 {
    var result: U128;
    var borrow = 0u;

    let diff0 = a.lo - b.lo;
    result.lo = diff0;
    borrow = select(0u, 1u, a.lo < b.lo);

    let diff1 = a.lo_hi - b.lo_hi - borrow;
    result.lo_hi = diff1;
    borrow = select(0u, 1u, a.lo_hi < b.lo_hi + borrow);

    let diff2 = a.hi_lo - b.hi_lo - borrow;
    result.hi_lo = diff2;
    borrow = select(0u, 1u, a.hi_lo < b.hi_lo + borrow);

    result.hi = a.hi - b.hi - borrow;

    return result;
}

// Multiply 64-bit by 64-bit -> 128-bit (simplified using 32-bit components)
fn u64_mul(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> U128 {
    // Karatsuba-style multiplication
    let ll = u32(a_lo) * u32(b_lo);
    let lh = u32(a_lo) * u32(b_hi);
    let hl = u32(a_hi) * u32(b_lo);
    let hh = u32(a_hi) * u32(b_hi);

    // Combine partial products
    var result: U128;
    result.lo = ll;

    let mid1 = lh + hl;
    let mid_carry = select(0u, 1u, mid1 < lh);

    // Add mid to result
    let sum1 = result.lo + (mid1 << 16u);  // Simplified shift
    result.lo = sum1;

    result.lo_hi = (mid1 >> 16u) + hh + select(0u, 1u, sum1 < result.lo);
    result.hi_lo = mid_carry;
    result.hi = 0u;

    return result;
}

// Right shift 128-bit value
fn u128_shr(a: U128, shift: u32) -> U128 {
    if (shift >= 128u) { return u128_zero(); }
    if (shift == 0u) { return a; }

    var result: U128;

    if (shift >= 96u) {
        result.lo = a.hi >> (shift - 96u);
        result.lo_hi = 0u;
        result.hi_lo = 0u;
        result.hi = 0u;
    } else if (shift >= 64u) {
        let s = shift - 64u;
        result.lo = (a.hi_lo >> s) | (a.hi << (32u - s));
        result.lo_hi = a.hi >> s;
        result.hi_lo = 0u;
        result.hi = 0u;
    } else if (shift >= 32u) {
        let s = shift - 32u;
        result.lo = (a.lo_hi >> s) | (a.hi_lo << (32u - s));
        result.lo_hi = (a.hi_lo >> s) | (a.hi << (32u - s));
        result.hi_lo = a.hi >> s;
        result.hi = 0u;
    } else {
        result.lo = (a.lo >> shift) | (a.lo_hi << (32u - shift));
        result.lo_hi = (a.lo_hi >> shift) | (a.hi_lo << (32u - shift));
        result.hi_lo = (a.hi_lo >> shift) | (a.hi << (32u - shift));
        result.hi = a.hi >> shift;
    }

    return result;
}

// Divide 128-bit by 64-bit (approximate for swap math)
fn u128_div_u64(n: U128, d_lo: u32, d_hi: u32) -> U128 {
    if (d_lo == 0u && d_hi == 0u) {
        return U128(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    }

    // Use floating point approximation for speed
    let n_f = f32(n.hi) * 79228162514264337593543950336.0 +
              f32(n.hi_lo) * 18446744073709551616.0 +
              f32(n.lo_hi) * 4294967296.0 +
              f32(n.lo);
    let d_f = f32(d_hi) * 4294967296.0 + f32(d_lo);

    let q = n_f / d_f;
    return U128(u32(q), u32(q / 4294967296.0), 0u, 0u);
}

// ============================================================================
// Swap Data Structures
// ============================================================================

struct SwapInput {
    // Pool identification
    pool_id_0: u32,
    pool_id_1: u32,
    pool_id_2: u32,
    pool_id_3: u32,

    // Current sqrt price (Q64.96)
    sqrt_price_lo: u32,
    sqrt_price_lo_hi: u32,
    sqrt_price_hi_lo: u32,
    sqrt_price_hi: u32,

    // Current liquidity
    liquidity_lo: u32,
    liquidity_hi: u32,

    // Current tick
    tick: i32,

    // Swap parameters
    zero_for_one: u32,   // Direction
    exact_input: u32,    // Amount type

    // Amount
    amount_lo: u32,
    amount_hi: u32,

    // Fee in pips (1 pip = 0.0001%)
    fee_pips: u32,

    // Price limit (Q64.96)
    sqrt_price_limit_lo: u32,
    sqrt_price_limit_hi: u32,
}

struct SwapOutput {
    // Amount deltas (signed as two's complement)
    amount0_delta_lo: u32,
    amount0_delta_hi: u32,
    amount1_delta_lo: u32,
    amount1_delta_hi: u32,

    // New sqrt price (Q64.96)
    sqrt_price_lo: u32,
    sqrt_price_hi: u32,

    // New tick
    tick: i32,

    // Success flag
    success: u32,

    // Fee growth increment
    fee_growth_lo: u32,
    fee_growth_hi: u32,

    // Error code (0 = success)
    error_code: u32,
}

// ============================================================================
// Liquidity Structures
// ============================================================================

struct LiquidityInput {
    pool_id_0: u32,
    pool_id_1: u32,

    sqrt_price_lo: u32,
    sqrt_price_hi: u32,

    liquidity_lo: u32,
    liquidity_hi: u32,

    current_tick: i32,
    tick_lower: i32,
    tick_upper: i32,

    is_add: u32,

    liquidity_delta_lo: u32,
    liquidity_delta_hi: u32,
}

struct LiquidityOutput {
    amount0_lo: u32,
    amount0_hi: u32,
    amount1_lo: u32,
    amount1_hi: u32,

    success: u32,
    error_code: u32,
}

// ============================================================================
// Parameter Structures
// ============================================================================

struct SwapParams {
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// ============================================================================
// Storage Bindings
// ============================================================================

@group(0) @binding(0) var<storage, read> swap_inputs: array<SwapInput>;
@group(0) @binding(1) var<storage, read_write> swap_outputs: array<SwapOutput>;
@group(0) @binding(2) var<uniform> params: SwapParams;

@group(0) @binding(3) var<storage, read> liquidity_inputs: array<LiquidityInput>;
@group(0) @binding(4) var<storage, read_write> liquidity_outputs: array<LiquidityOutput>;

// ============================================================================
// Core Math Functions
// ============================================================================

// Apply fee: amount * (1e6 - fee) / 1e6
fn apply_fee(amount_lo: u32, amount_hi: u32, fee_pips: u32) -> vec2<u32> {
    if (fee_pips == 0u) { return vec2<u32>(amount_lo, amount_hi); }

    let multiplier = FEE_DENOMINATOR - fee_pips;

    // Simplified: amount * multiplier / 1e6
    let product = u64_mul(amount_lo, amount_hi, multiplier, 0u);
    let result = u128_div_u64(product, FEE_DENOMINATOR, 0u);

    return vec2<u32>(result.lo, result.lo_hi);
}

// Calculate fee amount
fn calculate_fee(amount_lo: u32, amount_hi: u32, fee_pips: u32) -> vec2<u32> {
    if (fee_pips == 0u) { return vec2<u32>(0u, 0u); }

    let product = u64_mul(amount_lo, amount_hi, fee_pips, 0u);
    let result = u128_div_u64(product, FEE_DENOMINATOR, 0u);

    return vec2<u32>(result.lo, result.lo_hi);
}

// Calculate swap output: out = in * L / (L + in)
fn calculate_swap_output(
    amount_in_lo: u32, amount_in_hi: u32,
    liquidity_lo: u32, liquidity_hi: u32
) -> vec2<u32> {
    if (liquidity_lo == 0u && liquidity_hi == 0u) {
        return vec2<u32>(0u, 0u);
    }

    // numerator = amount_in * liquidity
    let numerator = u64_mul(amount_in_lo, amount_in_hi, liquidity_lo, liquidity_hi);

    // denominator = liquidity + amount_in
    let denom = u128_add(
        u128_from_u64(liquidity_lo, liquidity_hi),
        u128_from_u64(amount_in_lo, amount_in_hi)
    );

    if (u128_is_zero(denom)) {
        return vec2<u32>(0u, 0u);
    }

    let result = u128_div_u64(numerator, denom.lo, denom.lo_hi);
    return vec2<u32>(result.lo, result.lo_hi);
}

// ============================================================================
// Tick Math
// ============================================================================

// Convert tick to approximate sqrt price (simplified)
fn tick_to_sqrt_price(tick: i32) -> vec2<u32> {
    // Q96 = 2^96 at tick 0
    if (tick == 0) {
        return vec2<u32>(0u, 1u << 16u);  // Approximate 2^96
    }

    let abs_tick = select(-tick, tick, tick >= 0);

    // Use lookup table for common ticks (simplified)
    // In production, use precomputed magic numbers
    var ratio_lo = 1u << 16u;
    var ratio_hi = 1u;

    // Each tick is sqrt(1.0001) multiplier
    // Approximate using binary representation

    if (tick > 0) {
        // Price increases
        ratio_lo = ratio_lo + u32(abs_tick) * 50u;  // Very rough approximation
    } else {
        // Price decreases
        ratio_lo = ratio_lo - u32(abs_tick) * 50u;
    }

    return vec2<u32>(ratio_lo, ratio_hi);
}

// Convert sqrt price to tick (simplified)
fn sqrt_price_to_tick(sqrt_price_lo: u32, sqrt_price_hi: u32) -> i32 {
    // Simplified: find closest tick
    // In production, use log2 approximation

    let q96_lo = 0u;
    let q96_hi = 1u << 16u;  // Approximate 2^96

    if (sqrt_price_hi > q96_hi) {
        return i32((sqrt_price_hi - q96_hi) / 50u);
    } else if (sqrt_price_hi < q96_hi) {
        return -i32((q96_hi - sqrt_price_hi) / 50u);
    }

    return 0;
}

// ============================================================================
// Main Swap Kernel
// ============================================================================

@compute @workgroup_size(256)
fn batch_swap(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.count) { return; }

    let input = swap_inputs[gid.x];
    var output: SwapOutput;

    // Initialize output
    output.success = 1u;
    output.error_code = 0u;

    // Get input values
    let liquidity = vec2<u32>(input.liquidity_lo, input.liquidity_hi);
    let amount = vec2<u32>(input.amount_lo, input.amount_hi);
    let zero_for_one = input.zero_for_one != 0u;
    let exact_input = input.exact_input != 0u;

    // Check for zero liquidity
    if (liquidity.x == 0u && liquidity.y == 0u) {
        output.success = 0u;
        output.error_code = 1u;  // Zero liquidity
        swap_outputs[gid.x] = output;
        return;
    }

    // Apply fee to get effective amount
    let amount_after_fee = apply_fee(amount.x, amount.y, input.fee_pips);

    // Calculate swap amounts
    var amount0 = vec2<u32>(0u, 0u);
    var amount1 = vec2<u32>(0u, 0u);

    if (zero_for_one) {
        // Swapping token0 for token1
        if (exact_input) {
            amount0 = amount;
            amount1 = calculate_swap_output(
                amount_after_fee.x, amount_after_fee.y,
                liquidity.x, liquidity.y
            );
        } else {
            amount1 = amount;
            // For exact output, estimate required input
            amount0 = calculate_swap_output(
                amount.x, amount.y,
                liquidity.x, liquidity.y
            );
        }
    } else {
        // Swapping token1 for token0
        if (exact_input) {
            amount1 = amount;
            amount0 = calculate_swap_output(
                amount_after_fee.x, amount_after_fee.y,
                liquidity.x, liquidity.y
            );
        } else {
            amount0 = amount;
            amount1 = calculate_swap_output(
                amount.x, amount.y,
                liquidity.x, liquidity.y
            );
        }
    }

    // Calculate fee growth
    let fee_amount = select(
        calculate_fee(amount1.x, amount1.y, input.fee_pips),
        calculate_fee(amount0.x, amount0.y, input.fee_pips),
        zero_for_one
    );

    // Store outputs
    if (zero_for_one) {
        output.amount0_delta_lo = amount0.x;
        output.amount0_delta_hi = amount0.y;
        // amount1 is negative (output to user)
        output.amount1_delta_lo = ~amount1.x + 1u;
        output.amount1_delta_hi = select(~amount1.y + 1u, ~amount1.y, amount1.x == 0u);
    } else {
        output.amount1_delta_lo = amount1.x;
        output.amount1_delta_hi = amount1.y;
        output.amount0_delta_lo = ~amount0.x + 1u;
        output.amount0_delta_hi = select(~amount0.y + 1u, ~amount0.y, amount0.x == 0u);
    }

    // Update sqrt price (simplified)
    output.sqrt_price_lo = input.sqrt_price_lo;  // Would calculate new price
    output.sqrt_price_hi = input.sqrt_price_lo_hi;
    output.tick = input.tick;  // Would calculate new tick

    output.fee_growth_lo = fee_amount.x;
    output.fee_growth_hi = fee_amount.y;

    swap_outputs[gid.x] = output;
}

// ============================================================================
// Liquidity Modification Kernel
// ============================================================================

@compute @workgroup_size(256)
fn batch_liquidity(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.count) { return; }

    let input = liquidity_inputs[gid.x];
    var output: LiquidityOutput;

    output.success = 1u;
    output.error_code = 0u;

    // Check tick range
    if (input.tick_lower >= input.tick_upper) {
        output.success = 0u;
        output.error_code = 1u;  // Invalid tick range
        liquidity_outputs[gid.x] = output;
        return;
    }

    let current_tick = input.current_tick;
    let is_active = input.tick_lower <= current_tick && current_tick < input.tick_upper;

    let liq_delta = vec2<u32>(input.liquidity_delta_lo, input.liquidity_delta_hi);
    var amount0 = vec2<u32>(0u, 0u);
    var amount1 = vec2<u32>(0u, 0u);

    if (input.is_add != 0u) {
        // Adding liquidity
        if (is_active) {
            // Both tokens needed (split equally - simplified)
            amount0.x = liq_delta.x >> 1u;
            amount0.y = liq_delta.y >> 1u;
            amount1 = amount0;
        } else if (current_tick < input.tick_lower) {
            // Only token0 needed
            amount0 = liq_delta;
        } else {
            // Only token1 needed
            amount1 = liq_delta;
        }
    } else {
        // Removing liquidity (return as negative)
        if (is_active) {
            let half_x = liq_delta.x >> 1u;
            let half_y = liq_delta.y >> 1u;
            amount0.x = ~half_x + 1u;
            amount0.y = select(~half_y + 1u, ~half_y, half_x == 0u);
            amount1 = amount0;
        } else if (current_tick < input.tick_lower) {
            amount0.x = ~liq_delta.x + 1u;
            amount0.y = select(~liq_delta.y + 1u, ~liq_delta.y, liq_delta.x == 0u);
        } else {
            amount1.x = ~liq_delta.x + 1u;
            amount1.y = select(~liq_delta.y + 1u, ~liq_delta.y, liq_delta.x == 0u);
        }
    }

    output.amount0_lo = amount0.x;
    output.amount0_hi = amount0.y;
    output.amount1_lo = amount1.x;
    output.amount1_hi = amount1.y;

    liquidity_outputs[gid.x] = output;
}

// ============================================================================
// Route Optimization Kernel
// ============================================================================

struct RouteInput {
    amount_in_lo: u32,
    amount_in_hi: u32,

    // Up to 4 hops (simplified from 8 in Metal)
    sqrt_prices: array<vec2<u32>, 4>,
    liquidities: array<vec2<u32>, 4>,
    fees: array<u32, 4>,

    num_hops: u32,
}

struct RouteOutput {
    amount_out_lo: u32,
    amount_out_hi: u32,
    price_impact_bps: u32,
    success: u32,
    gas_estimate: u32,
}

@group(1) @binding(0) var<storage, read> route_inputs: array<RouteInput>;
@group(1) @binding(1) var<storage, read_write> route_outputs: array<RouteOutput>;

@compute @workgroup_size(256)
fn batch_route(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.count) { return; }

    let input = route_inputs[gid.x];
    var output: RouteOutput;
    output.success = 1u;

    if (input.num_hops == 0u) {
        output.success = 0u;
        output.amount_out_lo = 0u;
        output.amount_out_hi = 0u;
        route_outputs[gid.x] = output;
        return;
    }

    var current_amount = vec2<u32>(input.amount_in_lo, input.amount_in_hi);
    let initial_amount = current_amount;

    for (var i = 0u; i < input.num_hops && i < 4u; i++) {
        let liquidity = input.liquidities[i];
        let fee = input.fees[i];

        if (liquidity.x == 0u && liquidity.y == 0u) {
            output.success = 0u;
            break;
        }

        // Apply fee
        let amount_after_fee = apply_fee(current_amount.x, current_amount.y, fee);

        // Calculate output for this hop
        current_amount = calculate_swap_output(
            amount_after_fee.x, amount_after_fee.y,
            liquidity.x, liquidity.y
        );
    }

    output.amount_out_lo = current_amount.x;
    output.amount_out_hi = current_amount.y;

    // Calculate price impact
    if (initial_amount.x > 0u && current_amount.x > 0u) {
        var diff_lo = 0u;
        var diff_hi = 0u;

        if (initial_amount.x > current_amount.x ||
            (initial_amount.x == current_amount.x && initial_amount.y > current_amount.y)) {
            diff_lo = initial_amount.x - current_amount.x;
            diff_hi = initial_amount.y - current_amount.y;
        }

        // impact = diff / initial * 10000
        let impact_num = u64_mul(diff_lo, diff_hi, 10000u, 0u);
        let impact_result = u128_div_u64(impact_num, initial_amount.x, initial_amount.y);
        output.price_impact_bps = min(impact_result.lo, 10000u);
    } else {
        output.price_impact_bps = 0u;
    }

    // Gas estimate: ~30k per hop
    output.gas_estimate = input.num_hops * 30000u;

    route_outputs[gid.x] = output;
}

// ============================================================================
// Tick-to-SqrtPrice Batch Conversion
// ============================================================================

struct TickInput {
    tick: i32,
    _pad: i32,
}

struct TickOutput {
    sqrt_price_lo: u32,
    sqrt_price_hi: u32,
}

@group(2) @binding(0) var<storage, read> tick_inputs: array<TickInput>;
@group(2) @binding(1) var<storage, read_write> tick_outputs: array<TickOutput>;

@compute @workgroup_size(256)
fn batch_tick_to_sqrt_price(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.count) { return; }

    let tick = tick_inputs[gid.x].tick;
    let sqrt_price = tick_to_sqrt_price(tick);

    tick_outputs[gid.x].sqrt_price_lo = sqrt_price.x;
    tick_outputs[gid.x].sqrt_price_hi = sqrt_price.y;
}

// ============================================================================
// Price Impact Calculation Kernel
// ============================================================================

struct PriceImpactInput {
    amount_in_lo: u32,
    amount_in_hi: u32,
    amount_out_lo: u32,
    amount_out_hi: u32,
    spot_price_lo: u32,
    spot_price_hi: u32,
}

struct PriceImpactOutput {
    impact_bps: u32,
    _pad: u32,
}

@group(3) @binding(0) var<storage, read> impact_inputs: array<PriceImpactInput>;
@group(3) @binding(1) var<storage, read_write> impact_outputs: array<PriceImpactOutput>;

@compute @workgroup_size(256)
fn batch_price_impact(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= params.count) { return; }

    let input = impact_inputs[gid.x];

    let amount_in = vec2<u32>(input.amount_in_lo, input.amount_in_hi);
    let amount_out = vec2<u32>(input.amount_out_lo, input.amount_out_hi);
    let spot_price = vec2<u32>(input.spot_price_lo, input.spot_price_hi);

    if (amount_in.x == 0u && amount_in.y == 0u) {
        impact_outputs[gid.x].impact_bps = 0u;
        return;
    }

    // Expected output at spot price (simplified)
    let expected = u64_mul(amount_in.x, amount_in.y, spot_price.x, spot_price.y);
    let expected_out = u128_shr(expected, Q96_BITS);

    // Impact = (expected - actual) / expected * 10000
    var diff: U128;
    if (u128_gt(expected_out, u128_from_u64(amount_out.x, amount_out.y))) {
        diff = u128_sub(expected_out, u128_from_u64(amount_out.x, amount_out.y));
    } else {
        impact_outputs[gid.x].impact_bps = 0u;
        return;
    }

    let impact_num = u64_mul(diff.lo, diff.lo_hi, 10000u, 0u);
    let impact = u128_div_u64(impact_num, expected_out.lo, expected_out.lo_hi);

    impact_outputs[gid.x].impact_bps = min(impact.lo, 10000u);
}

// ============================================================================
// Batch Quote Kernel (get quote without executing)
// ============================================================================

@compute @workgroup_size(256)
fn batch_quote(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // Same as batch_swap but doesn't modify state
    if (gid.x >= params.count) { return; }

    let input = swap_inputs[gid.x];
    var output: SwapOutput;

    output.success = 1u;
    output.error_code = 0u;

    let liquidity = vec2<u32>(input.liquidity_lo, input.liquidity_hi);
    let amount = vec2<u32>(input.amount_lo, input.amount_hi);

    if (liquidity.x == 0u && liquidity.y == 0u) {
        output.success = 0u;
        output.error_code = 1u;
        swap_outputs[gid.x] = output;
        return;
    }

    let amount_after_fee = apply_fee(amount.x, amount.y, input.fee_pips);
    let out_amount = calculate_swap_output(
        amount_after_fee.x, amount_after_fee.y,
        liquidity.x, liquidity.y
    );

    if (input.zero_for_one != 0u) {
        output.amount0_delta_lo = amount.x;
        output.amount0_delta_hi = amount.y;
        output.amount1_delta_lo = out_amount.x;
        output.amount1_delta_hi = out_amount.y;
    } else {
        output.amount1_delta_lo = amount.x;
        output.amount1_delta_hi = amount.y;
        output.amount0_delta_lo = out_amount.x;
        output.amount0_delta_hi = out_amount.y;
    }

    output.sqrt_price_lo = input.sqrt_price_lo;
    output.sqrt_price_hi = input.sqrt_price_lo_hi;
    output.tick = input.tick;
    output.fee_growth_lo = 0u;
    output.fee_growth_hi = 0u;

    swap_outputs[gid.x] = output;
}
