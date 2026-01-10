// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// DEX Swap Acceleration Kernels for WebGPU
// Implements high-performance Uniswap v4-style AMM math
// Cross-platform GPU acceleration via WebGPU/WGSL

// =============================================================================
// 128-bit Unsigned Integer Type (as vec2<u64> simulation using vec4<u32>)
// =============================================================================

// Since WGSL doesn't have native u64, we use vec4<u32> for 128-bit
// [lo_lo, lo_hi, hi_lo, hi_hi] = 4 x 32-bit words

fn u128_from_u64(lo: u32, hi: u32) -> vec4<u32> {
    return vec4<u32>(lo, 0u, hi, 0u);
}

fn u128_from_parts(lo_lo: u32, lo_hi: u32, hi_lo: u32, hi_hi: u32) -> vec4<u32> {
    return vec4<u32>(lo_lo, lo_hi, hi_lo, hi_hi);
}

fn u128_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var result: vec4<u32>;
    var carry: u32 = 0u;
    
    // Add word 0
    let sum0 = u64(a.x) + u64(b.x) + u64(carry);
    result.x = u32(sum0);
    carry = u32(sum0 >> 32u);
    
    // Add word 1
    let sum1 = u64(a.y) + u64(b.y) + u64(carry);
    result.y = u32(sum1);
    carry = u32(sum1 >> 32u);
    
    // Add word 2
    let sum2 = u64(a.z) + u64(b.z) + u64(carry);
    result.z = u32(sum2);
    carry = u32(sum2 >> 32u);
    
    // Add word 3
    let sum3 = u64(a.w) + u64(b.w) + u64(carry);
    result.w = u32(sum3);
    
    return result;
}

fn u128_sub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var result: vec4<u32>;
    var borrow: u32 = 0u;
    
    // Subtract word 0
    let diff0 = i64(a.x) - i64(b.x) - i64(borrow);
    if (diff0 < 0) {
        result.x = u32(diff0 + 0x100000000);
        borrow = 1u;
    } else {
        result.x = u32(diff0);
        borrow = 0u;
    }
    
    // Subtract word 1
    let diff1 = i64(a.y) - i64(b.y) - i64(borrow);
    if (diff1 < 0) {
        result.y = u32(diff1 + 0x100000000);
        borrow = 1u;
    } else {
        result.y = u32(diff1);
        borrow = 0u;
    }
    
    // Subtract word 2
    let diff2 = i64(a.z) - i64(b.z) - i64(borrow);
    if (diff2 < 0) {
        result.z = u32(diff2 + 0x100000000);
        borrow = 1u;
    } else {
        result.z = u32(diff2);
        borrow = 0u;
    }
    
    // Subtract word 3
    let diff3 = i64(a.w) - i64(b.w) - i64(borrow);
    if (diff3 < 0) {
        result.w = u32(diff3 + 0x100000000);
    } else {
        result.w = u32(diff3);
    }
    
    return result;
}

fn u128_is_zero(v: vec4<u32>) -> bool {
    return v.x == 0u && v.y == 0u && v.z == 0u && v.w == 0u;
}

fn u128_lt(a: vec4<u32>, b: vec4<u32>) -> bool {
    if (a.w != b.w) { return a.w < b.w; }
    if (a.z != b.z) { return a.z < b.z; }
    if (a.y != b.y) { return a.y < b.y; }
    return a.x < b.x;
}

fn u128_gt(a: vec4<u32>, b: vec4<u32>) -> bool {
    return u128_lt(b, a);
}

// Multiply 32x32 -> 64
fn mul32x32(a: u32, b: u32) -> vec2<u32> {
    let product = u64(a) * u64(b);
    return vec2<u32>(u32(product), u32(product >> 32u));
}

// Multiply two 64-bit numbers (as vec2<u32>) -> 128-bit (as vec4<u32>)
fn mul64x64(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    // a = a.x + a.y * 2^32
    // b = b.x + b.y * 2^32
    // product = a.x*b.x + (a.x*b.y + a.y*b.x)*2^32 + a.y*b.y*2^64
    
    let ll = mul32x32(a.x, b.x);
    let lh = mul32x32(a.x, b.y);
    let hl = mul32x32(a.y, b.x);
    let hh = mul32x32(a.y, b.y);
    
    var result: vec4<u32>;
    result.x = ll.x;
    
    // Add mid products at position 32
    let mid_lo = u64(ll.y) + u64(lh.x) + u64(hl.x);
    result.y = u32(mid_lo);
    let carry1 = u32(mid_lo >> 32u);
    
    // Add high products at position 64
    let mid_hi = u64(lh.y) + u64(hl.y) + u64(hh.x) + u64(carry1);
    result.z = u32(mid_hi);
    let carry2 = u32(mid_hi >> 32u);
    
    result.w = hh.y + carry2;
    
    return result;
}

// Divide 128-bit by 32-bit (approximate)
fn u128_div_u32(n: vec4<u32>, d: u32) -> vec4<u32> {
    if (d == 0u) { return vec4<u32>(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu); }
    
    // Use floating point approximation
    let n_f = f32(n.w) * 79228162514264337593543950336.0 + 
              f32(n.z) * 18446744073709551616.0 +
              f32(n.y) * 4294967296.0 +
              f32(n.x);
    let result_f = n_f / f32(d);
    
    // Convert back (lossy but fast)
    let result = u32(result_f);
    return vec4<u32>(result, 0u, 0u, 0u);
}

// Right shift 128-bit
fn u128_rsh(v: vec4<u32>, shift: u32) -> vec4<u32> {
    if (shift >= 128u) { return vec4<u32>(0u, 0u, 0u, 0u); }
    if (shift == 0u) { return v; }
    
    if (shift >= 96u) {
        return vec4<u32>(v.w >> (shift - 96u), 0u, 0u, 0u);
    }
    if (shift >= 64u) {
        let s = shift - 64u;
        return vec4<u32>(
            (v.z >> s) | (v.w << (32u - s)),
            v.w >> s,
            0u,
            0u
        );
    }
    if (shift >= 32u) {
        let s = shift - 32u;
        return vec4<u32>(
            (v.y >> s) | (v.z << (32u - s)),
            (v.z >> s) | (v.w << (32u - s)),
            v.w >> s,
            0u
        );
    }
    
    // shift < 32
    return vec4<u32>(
        (v.x >> shift) | (v.y << (32u - shift)),
        (v.y >> shift) | (v.z << (32u - shift)),
        (v.z >> shift) | (v.w << (32u - shift)),
        v.w >> shift
    );
}

// =============================================================================
// Swap Input/Output Structures
// =============================================================================

struct SwapInput {
    poolId: vec4<u32>,           // Pool ID (128 bits as 4x32)
    sqrtPriceX96: vec4<u32>,     // sqrt price (128 bits)
    liquidity: vec4<u32>,        // liquidity (128 bits)
    tick: i32,
    zeroForOne: u32,             // bool
    exactInput: u32,             // bool
    feePips: u32,
    amount: vec4<u32>,           // amount (128 bits)
    sqrtPriceLimit: vec4<u32>,   // price limit (128 bits)
}

struct SwapOutput {
    amount0Delta: vec4<u32>,     // token0 delta (128 bits, signed as 2's complement)
    amount1Delta: vec4<u32>,     // token1 delta
    sqrtPriceX96: vec4<u32>,     // new sqrt price
    tick: i32,
    success: u32,
    feeGrowth: vec4<u32>,
    errorCode: u32,
    padding: u32,
}

struct LiquidityInput {
    poolId: vec4<u32>,
    sqrtPriceX96: vec4<u32>,
    liquidity: vec4<u32>,
    currentTick: i32,
    tickLower: i32,
    tickUpper: i32,
    isAdd: u32,
    liqDelta: vec4<u32>,
}

struct LiquidityOutput {
    amount0: vec4<u32>,
    amount1: vec4<u32>,
    feeGrowth0: vec4<u32>,
    feeGrowth1: vec4<u32>,
    success: u32,
    errorCode: u32,
    padding: vec2<u32>,
}

struct RouteInput {
    amountIn: vec4<u32>,
    sqrtPrices: array<vec4<u32>, 8>,
    liquidities: array<vec4<u32>, 8>,
    fees: array<u32, 8>,
    numHops: u32,
    padding: array<u32, 3>,
}

struct RouteOutput {
    amountOut: vec4<u32>,
    priceImpact: u32,
    success: u32,
    gasEstimate: u32,
    padding: u32,
}

// =============================================================================
// Bindings
// =============================================================================

@group(0) @binding(0) var<storage, read> swap_inputs: array<SwapInput>;
@group(0) @binding(1) var<storage, read_write> swap_outputs: array<SwapOutput>;
@group(0) @binding(2) var<uniform> swap_count: u32;

@group(1) @binding(0) var<storage, read> liq_inputs: array<LiquidityInput>;
@group(1) @binding(1) var<storage, read_write> liq_outputs: array<LiquidityOutput>;
@group(1) @binding(2) var<uniform> liq_count: u32;

@group(2) @binding(0) var<storage, read> route_inputs: array<RouteInput>;
@group(2) @binding(1) var<storage, read_write> route_outputs: array<RouteOutput>;
@group(2) @binding(2) var<uniform> route_count: u32;

// =============================================================================
// Core Math Functions
// =============================================================================

fn apply_fee(amount: vec4<u32>, fee_pips: u32) -> vec4<u32> {
    if (fee_pips == 0u) { return amount; }
    
    let multiplier = 1000000u - fee_pips;
    
    // Multiply by multiplier (simplified for 64-bit amounts)
    let amount_lo = vec2<u32>(amount.x, amount.y);
    let result = mul64x64(amount_lo, vec2<u32>(multiplier, 0u));
    
    // Divide by 1e6
    return u128_div_u32(result, 1000000u);
}

fn calculate_fee(amount: vec4<u32>, fee_pips: u32) -> vec4<u32> {
    if (fee_pips == 0u) { return vec4<u32>(0u, 0u, 0u, 0u); }
    
    let amount_lo = vec2<u32>(amount.x, amount.y);
    let fee = mul64x64(amount_lo, vec2<u32>(fee_pips, 0u));
    
    return u128_div_u32(fee, 1000000u);
}

fn calculate_swap_output(amount_in: vec4<u32>, liquidity: vec4<u32>) -> vec4<u32> {
    if (u128_is_zero(liquidity)) { return vec4<u32>(0u, 0u, 0u, 0u); }
    
    // out = in * L / (L + in)
    let in_lo = vec2<u32>(amount_in.x, amount_in.y);
    let liq_lo = vec2<u32>(liquidity.x, liquidity.y);
    
    let numerator = mul64x64(in_lo, liq_lo);
    let denominator = u128_add(liquidity, amount_in);
    
    if (u128_is_zero(denominator)) { return vec4<u32>(0u, 0u, 0u, 0u); }
    
    // Approximate division
    return u128_div_u32(numerator, denominator.x);
}

fn negate_u128(v: vec4<u32>) -> vec4<u32> {
    // 2's complement negation
    let inverted = vec4<u32>(~v.x, ~v.y, ~v.z, ~v.w);
    return u128_add(inverted, vec4<u32>(1u, 0u, 0u, 0u));
}

fn sqrt_price_to_tick(sqrt_price: vec4<u32>) -> i32 {
    // Q96 reference (2^96)
    let q96 = vec4<u32>(0u, 0x100000000u, 0u, 0u);
    
    if (sqrt_price.x == q96.x && sqrt_price.y == q96.y) { return 0; }
    if (u128_lt(sqrt_price, q96)) { return -1; }
    return 1;
}

// =============================================================================
// Main Swap Kernel
// =============================================================================

@compute @workgroup_size(64)
fn batch_swap(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = id.x;
    if (tid >= swap_count) { return; }
    
    let input = swap_inputs[tid];
    var output: SwapOutput;
    
    output.success = 1u;
    output.errorCode = 0u;
    
    let sqrt_price = input.sqrtPriceX96;
    let liquidity = input.liquidity;
    let amount = input.amount;
    let zero_for_one = input.zeroForOne != 0u;
    let exact_input = input.exactInput != 0u;
    
    // Check zero liquidity
    if (u128_is_zero(liquidity)) {
        output.success = 0u;
        output.errorCode = 1u;
        swap_outputs[tid] = output;
        return;
    }
    
    // Apply fee
    let amount_after_fee = apply_fee(amount, input.feePips);
    
    var amount0: vec4<u32>;
    var amount1: vec4<u32>;
    
    if (zero_for_one) {
        if (exact_input) {
            amount0 = amount;
            amount1 = calculate_swap_output(amount_after_fee, liquidity);
        } else {
            amount1 = amount;
            // Calculate input for exact output
            if (u128_gt(liquidity, amount)) {
                let denom = u128_sub(liquidity, amount);
                let in_lo = vec2<u32>(amount.x, amount.y);
                let liq_lo = vec2<u32>(liquidity.x, liquidity.y);
                let numer = mul64x64(in_lo, liq_lo);
                amount0 = u128_div_u32(numer, denom.x);
            } else {
                amount0 = liquidity;
            }
        }
    } else {
        if (exact_input) {
            amount1 = amount;
            amount0 = calculate_swap_output(amount_after_fee, liquidity);
        } else {
            amount0 = amount;
            if (u128_gt(liquidity, amount)) {
                let denom = u128_sub(liquidity, amount);
                let in_lo = vec2<u32>(amount.x, amount.y);
                let liq_lo = vec2<u32>(liquidity.x, liquidity.y);
                let numer = mul64x64(in_lo, liq_lo);
                amount1 = u128_div_u32(numer, denom.x);
            } else {
                amount1 = liquidity;
            }
        }
    }
    
    // Calculate new sqrt price (simplified)
    var new_sqrt_price = sqrt_price;
    if (!u128_is_zero(liquidity)) {
        let amount_lo = vec2<u32>(amount0.x, amount0.y);
        let price_lo = vec2<u32>(sqrt_price.x, sqrt_price.y);
        let delta = mul64x64(amount_lo, price_lo);
        let delta_scaled = u128_div_u32(delta, liquidity.x);
        
        if (zero_for_one) {
            if (u128_gt(sqrt_price, delta_scaled)) {
                new_sqrt_price = u128_sub(sqrt_price, delta_scaled);
            } else {
                new_sqrt_price = vec4<u32>(1u, 0u, 0u, 0u);
            }
        } else {
            new_sqrt_price = u128_add(sqrt_price, delta_scaled);
        }
    }
    
    // Fee growth
    var fee_src = amount0;
    if (!zero_for_one) { fee_src = amount1; }
    let fee_growth = calculate_fee(fee_src, input.feePips);
    
    // Set outputs with proper signs
    if (zero_for_one) {
        output.amount0Delta = amount0;
        output.amount1Delta = negate_u128(amount1);
    } else {
        output.amount1Delta = amount1;
        output.amount0Delta = negate_u128(amount0);
    }
    
    output.sqrtPriceX96 = new_sqrt_price;
    output.tick = sqrt_price_to_tick(new_sqrt_price);
    output.feeGrowth = fee_growth;
    
    swap_outputs[tid] = output;
}

// =============================================================================
// Liquidity Kernel
// =============================================================================

@compute @workgroup_size(64)
fn batch_liquidity(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = id.x;
    if (tid >= liq_count) { return; }
    
    let input = liq_inputs[tid];
    var output: LiquidityOutput;
    
    output.success = 1u;
    output.errorCode = 0u;
    
    // Check tick range
    if (input.tickLower >= input.tickUpper) {
        output.success = 0u;
        output.errorCode = 1u;
        liq_outputs[tid] = output;
        return;
    }
    
    let current_tick = input.currentTick;
    let is_active = input.tickLower <= current_tick && current_tick < input.tickUpper;
    let liq_delta = input.liqDelta;
    
    var amount0: vec4<u32>;
    var amount1: vec4<u32>;
    
    if (input.isAdd != 0u) {
        // Adding liquidity
        if (is_active) {
            amount0 = u128_rsh(liq_delta, 1u);
            amount1 = u128_rsh(liq_delta, 1u);
        } else if (current_tick < input.tickLower) {
            amount0 = liq_delta;
            amount1 = vec4<u32>(0u, 0u, 0u, 0u);
        } else {
            amount0 = vec4<u32>(0u, 0u, 0u, 0u);
            amount1 = liq_delta;
        }
    } else {
        // Removing liquidity
        let half = u128_rsh(liq_delta, 1u);
        if (is_active) {
            amount0 = negate_u128(half);
            amount1 = negate_u128(half);
        } else if (current_tick < input.tickLower) {
            amount0 = negate_u128(liq_delta);
            amount1 = vec4<u32>(0u, 0u, 0u, 0u);
        } else {
            amount0 = vec4<u32>(0u, 0u, 0u, 0u);
            amount1 = negate_u128(liq_delta);
        }
    }
    
    output.amount0 = amount0;
    output.amount1 = amount1;
    output.feeGrowth0 = vec4<u32>(0u, 0u, 0u, 0u);
    output.feeGrowth1 = vec4<u32>(0u, 0u, 0u, 0u);
    
    liq_outputs[tid] = output;
}

// =============================================================================
// Route Optimization Kernel
// =============================================================================

@compute @workgroup_size(64)
fn batch_route(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = id.x;
    if (tid >= route_count) { return; }
    
    let input = route_inputs[tid];
    var output: RouteOutput;
    
    output.success = 1u;
    
    if (input.numHops == 0u) {
        output.success = 0u;
        output.amountOut = vec4<u32>(0u, 0u, 0u, 0u);
        route_outputs[tid] = output;
        return;
    }
    
    var current_amount = input.amountIn;
    let initial_amount = current_amount;
    
    for (var i = 0u; i < input.numHops && i < 8u; i++) {
        let liquidity = input.liquidities[i];
        let fee = input.fees[i];
        
        if (u128_is_zero(liquidity)) {
            output.success = 0u;
            break;
        }
        
        // Apply fee and calculate output
        let amount_after_fee = apply_fee(current_amount, fee);
        current_amount = calculate_swap_output(amount_after_fee, liquidity);
    }
    
    output.amountOut = current_amount;
    
    // Calculate price impact
    if (!u128_is_zero(initial_amount) && !u128_is_zero(current_amount)) {
        if (u128_gt(initial_amount, current_amount)) {
            let diff = u128_sub(initial_amount, current_amount);
            let diff_lo = vec2<u32>(diff.x, diff.y);
            let impact_num = mul64x64(diff_lo, vec2<u32>(10000u, 0u));
            let impact = u128_div_u32(impact_num, initial_amount.x);
            output.priceImpact = min(impact.x, 10000u);
        } else {
            output.priceImpact = 0u;
        }
    } else {
        output.priceImpact = 0u;
    }
    
    // Gas estimate
    output.gasEstimate = input.numHops * 30000u;
    
    route_outputs[tid] = output;
}

// =============================================================================
// Tick Math Kernels
// =============================================================================

struct TickMathInput {
    tick: i32,
    padding: i32,
}

struct TickMathOutput {
    sqrtPriceX96: vec4<u32>,
}

@group(3) @binding(0) var<storage, read> tick_inputs: array<TickMathInput>;
@group(3) @binding(1) var<storage, read_write> tick_outputs: array<TickMathOutput>;
@group(3) @binding(2) var<uniform> tick_count: u32;

// Magic constants for sqrt(1.0001^(2^i))
const SQRT_MAGIC_0: u32 = 0xfff97263u;
const SQRT_MAGIC_1: u32 = 0xfff2e50fu;
const SQRT_MAGIC_2: u32 = 0xffe5cacau;
const SQRT_MAGIC_3: u32 = 0xffcb9a97u;
const SQRT_MAGIC_4: u32 = 0xff97383cu;

@compute @workgroup_size(64)
fn batch_tick_to_sqrt_price(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = id.x;
    if (tid >= tick_count) { return; }
    
    let tick = tick_inputs[tid].tick;
    
    // Q96 = 2^96
    if (tick == 0) {
        tick_outputs[tid].sqrtPriceX96 = vec4<u32>(0u, 0x100000000u, 0u, 0u);
        return;
    }
    
    var abs_tick: u32;
    if (tick < 0) {
        abs_tick = u32(-tick);
    } else {
        abs_tick = u32(tick);
    }
    
    // Start with Q128
    var ratio = vec4<u32>(0u, 0u, 0u, 1u);
    
    // Apply magic multipliers
    if ((abs_tick & 1u) != 0u) {
        ratio = mul64x64(vec2<u32>(ratio.x, ratio.y), vec2<u32>(SQRT_MAGIC_0, 0u));
        ratio = u128_rsh(ratio, 48u);
    }
    if ((abs_tick & 2u) != 0u) {
        ratio = mul64x64(vec2<u32>(ratio.x, ratio.y), vec2<u32>(SQRT_MAGIC_1, 0u));
        ratio = u128_rsh(ratio, 48u);
    }
    if ((abs_tick & 4u) != 0u) {
        ratio = mul64x64(vec2<u32>(ratio.x, ratio.y), vec2<u32>(SQRT_MAGIC_2, 0u));
        ratio = u128_rsh(ratio, 48u);
    }
    if ((abs_tick & 8u) != 0u) {
        ratio = mul64x64(vec2<u32>(ratio.x, ratio.y), vec2<u32>(SQRT_MAGIC_3, 0u));
        ratio = u128_rsh(ratio, 48u);
    }
    if ((abs_tick & 16u) != 0u) {
        ratio = mul64x64(vec2<u32>(ratio.x, ratio.y), vec2<u32>(SQRT_MAGIC_4, 0u));
        ratio = u128_rsh(ratio, 48u);
    }
    
    // Invert for negative ticks
    if (tick < 0) {
        let max_val = vec4<u32>(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
        ratio = u128_div_u32(max_val, ratio.x);
    }
    
    // Convert from Q128 to Q96
    ratio = u128_rsh(ratio, 32u);
    
    tick_outputs[tid].sqrtPriceX96 = ratio;
}

// =============================================================================
// Next Initialized Tick Search
// =============================================================================

struct TickSearchInput {
    currentTick: i32,
    tickSpacing: i32,
    searchLeft: u32,
    padding: u32,
    bitmapWords: array<vec4<u32>, 16>,  // 16 words, each 256 bits
    bitmapStartWord: i32,
    bitmapEndWord: i32,
    padding2: vec2<i32>,
}

struct TickSearchOutput {
    nextTick: i32,
    isInitialized: u32,
}

@group(4) @binding(0) var<storage, read> search_inputs: array<TickSearchInput>;
@group(4) @binding(1) var<storage, read_write> search_outputs: array<TickSearchOutput>;
@group(4) @binding(2) var<uniform> search_count: u32;

fn count_leading_zeros(v: u32) -> u32 {
    if (v == 0u) { return 32u; }
    var n = 0u;
    if (v <= 0x0000FFFFu) { n += 16u; }
    if (v <= 0x00FFFFFFu) { n += 8u; }
    if (v <= 0x0FFFFFFFu) { n += 4u; }
    if (v <= 0x3FFFFFFFu) { n += 2u; }
    if (v <= 0x7FFFFFFFu) { n += 1u; }
    return n;
}

fn count_trailing_zeros(v: u32) -> u32 {
    if (v == 0u) { return 32u; }
    var n = 0u;
    var x = v;
    if ((x & 0x0000FFFFu) == 0u) { n += 16u; x >>= 16u; }
    if ((x & 0x000000FFu) == 0u) { n += 8u; x >>= 8u; }
    if ((x & 0x0000000Fu) == 0u) { n += 4u; x >>= 4u; }
    if ((x & 0x00000003u) == 0u) { n += 2u; x >>= 2u; }
    if ((x & 0x00000001u) == 0u) { n += 1u; }
    return n;
}

@compute @workgroup_size(64)
fn batch_next_initialized_tick(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = id.x;
    if (tid >= search_count) { return; }
    
    let input = search_inputs[tid];
    var output: TickSearchOutput;
    output.isInitialized = 0u;
    
    let compressed = input.currentTick / input.tickSpacing;
    let wp = i32(compressed >> 8);
    let bp = u32(compressed & 0xFF);
    let search_left = input.searchLeft != 0u;
    
    if (search_left) {
        // Search toward lower ticks
        for (var search_wp = wp; search_wp >= input.bitmapStartWord; search_wp--) {
            let word_idx = search_wp - input.bitmapStartWord;
            if (word_idx < 0 || word_idx >= 16) { continue; }
            
            let word = input.bitmapWords[word_idx];
            
            // Search each 32-bit segment
            var start_seg = 3;
            if (search_wp == wp) { start_seg = i32(bp / 32u); }
            
            for (var seg = start_seg; seg >= 0; seg--) {
                var w: u32;
                switch(seg) {
                    case 0: { w = word.x; }
                    case 1: { w = word.y; }
                    case 2: { w = word.z; }
                    case 3: { w = word.w; }
                    default: { w = 0u; }
                }
                
                if (search_wp == wp && seg == i32(bp / 32u)) {
                    let mask = (1u << (bp % 32u + 1u)) - 1u;
                    w &= mask;
                }
                
                if (w != 0u) {
                    let high_bit = 31u - count_leading_zeros(w);
                    output.nextTick = (search_wp * 256 + seg * 32 + i32(high_bit)) * input.tickSpacing;
                    output.isInitialized = 1u;
                    search_outputs[tid] = output;
                    return;
                }
            }
        }
        
        output.nextTick = -887272 * input.tickSpacing;
    } else {
        // Search toward higher ticks
        for (var search_wp = wp; search_wp <= input.bitmapEndWord; search_wp++) {
            let word_idx = search_wp - input.bitmapStartWord;
            if (word_idx < 0 || word_idx >= 16) { continue; }
            
            let word = input.bitmapWords[word_idx];
            
            var start_seg = 0;
            var start_bit = 0u;
            if (search_wp == wp) {
                start_bit = bp + 1u;
                start_seg = i32(start_bit / 32u);
            }
            
            for (var seg = start_seg; seg < 4; seg++) {
                var w: u32;
                switch(seg) {
                    case 0: { w = word.x; }
                    case 1: { w = word.y; }
                    case 2: { w = word.z; }
                    case 3: { w = word.w; }
                    default: { w = 0u; }
                }
                
                if (search_wp == wp && seg == i32(start_bit / 32u)) {
                    let mask = ~((1u << (start_bit % 32u)) - 1u);
                    w &= mask;
                }
                
                if (w != 0u) {
                    let low_bit = count_trailing_zeros(w);
                    output.nextTick = (search_wp * 256 + seg * 32 + i32(low_bit)) * input.tickSpacing;
                    output.isInitialized = 1u;
                    search_outputs[tid] = output;
                    return;
                }
            }
        }
        
        output.nextTick = 887272 * input.tickSpacing;
    }
    
    search_outputs[tid] = output;
}
