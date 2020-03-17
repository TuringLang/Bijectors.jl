if VERSION < v"1.1"
    isnothing(x) = x === nothing
end

struct ADTestFunction
    name::String
    f::Function
    x::Vector
end
struct DistSpec{Tθ<:Tuple, Tx}
    name::Union{Symbol, Expr}
    θ::Tθ
    x::Tx
end

vectorize(v::Number) = [v]
vectorize(v::Diagonal) = v.diag
vectorize(v::Vector{<:Matrix}) = mapreduce(vec, vcat, v)
vectorize(v) = vec(v)
pack(vals...) = reduce(vcat, vectorize.(vals))

@generated function unpack(x, vals...)
    unpacked = []
    ind = :(1)
    for (i, T) in enumerate(vals)
        if T <: Number
            push!(unpacked, :(x[$ind]))
            ind = :($ind + 1)
        elseif T <: Vector{<:Matrix}
            push!(unpacked, :(unpack_vec_of_mats(x[$ind:$ind+sum(length, vals[$i])-1], vals[$i])))
            ind = :($ind + sum(length, vals[$i]))
        elseif T <: Vector
            push!(unpacked, :(x[$ind:$ind+length(vals[$i])-1]))
            ind = :($ind + length(vals[$i]))
        elseif T <: Diagonal
            push!(unpacked, :(Diagonal(x[$ind:$ind+size(vals[$i],1)-1])))
            ind = :($ind + size(vals[$i],1))
        elseif T <: Matrix
            push!(unpacked, :(reshape(x[$ind:($ind+length(vals[$i])-1)], size(vals[$i]))))
            ind = :($ind + length(vals[$i]))
        else
            throw("Unsupported argument")
        end
    end
    return quote
        @assert $ind == length(x) + 1
        return ($(unpacked...),)
    end
end
function unpack_vec_of_mats(x, val)
    ind = 1
    return map(1:length(val)) do i 
        out = reshape(x[ind : ind + length(val[i]) - 1], size(val[i]))        
        ind += length(val[i])
        out
    end
end

function get_function(dist::DistSpec, inds, val)
    syms = []
    args = []
    for (i, a) in enumerate(dist.θ)
        if i in inds
            sym = gensym()
            push!(syms, sym)
            push!(args, sym)
        else
            push!(args, a)
        end
    end
    if val
        sym = gensym()
        push!(syms, sym)
        expr = quote
            ($(syms...),) -> begin
                temp_args = ($(args...),)
                temp_dist = $(dist.name)(temp_args...)
                temp_x = $(sym)
                link(temp_dist, temp_x)
                temp = logpdf_with_trans(temp_dist, invlink(temp_dist, temp_x), true)
                if temp isa AbstractVector
                    return sum(temp)
                else
                    return temp
                end
            end
        end
        if length(inds) == 0
            f = eval(:(x -> $expr(unpack(x, $(dist.x))...)))
            return ADTestFunction(string(expr), f, pack(dist.x))
        else
            f = eval(:(x -> $expr(unpack(x, $(dist.θ[inds]...), $(dist.x))...)))
            return ADTestFunction(string(expr), f, pack(dist.θ[inds]..., dist.x))
        end
    else
        @assert length(inds) > 0
        expr = quote
            ($(syms...),) -> begin
                temp_args = ($(args...),)
                temp_dist = $(dist.name)(temp_args...)
                temp_x = $(dist.x)
                link(temp_dist, temp_x)
                temp = logpdf_with_trans(temp_dist, invlink(temp_dist, temp_x), true)
                if temp isa AbstractVector
                    return sum(temp)
                else
                    return temp
                end
            end
        end
        expr = :(x -> $expr(unpack(x, $(dist.θ[inds]...))...))
        f = eval(expr)
        return ADTestFunction(string(expr), f, pack(dist.θ[inds]...))
    end
end
function get_all_functions(dist::DistSpec, continuous=false)
    fs = []
    if length(dist.θ) == 0
        push!(fs, get_function(dist, (), true))
    else
        for inds in combinations(1:length(dist.θ))
            push!(fs, get_function(dist, inds, false))
            if continuous
                push!(fs, get_function(dist, inds, true))
            end
        end
    end
    return fs
end

const zygote_counter = Ref(0)

function test_ad(f, at = 0.5; rtol = 1e-8, atol = 1e-8)
    stg = get_stage()
    if stg == "all"
        reverse_tracker = Tracker.data(Tracker.gradient(f, at)[1])
        reverse_zygote = Zygote.gradient(f, at)[1]
        forward = ForwardDiff.gradient(f, at)
        reverse_diff = ReverseDiff.gradient(f, at)
        @test isapprox(reverse_tracker, forward, rtol=rtol, atol=atol)
        @test isapprox(reverse_zygote, forward, rtol=rtol, atol=atol)
        @test isapprox(reverse_diff, forward, rtol=rtol, atol=atol)
    elseif stg == "ForwardDiff_Tracker"
        reverse_tracker = Tracker.data(Tracker.gradient(f, at)[1])
        forward = ForwardDiff.gradient(f, at)
        @test isapprox(reverse_tracker, forward, rtol=rtol, atol=atol)
    elseif stg == "Zygote"
        zygote_counter[] += 1
        if mod(zygote_counter[], 30) == 0
            Zygote.refresh()
        end
        reverse_zygote = Zygote.gradient(f, at)[1]
        forward = ForwardDiff.gradient(f, at)
        @test isapprox(reverse_zygote, forward, rtol=rtol, atol=atol)
    elseif stg == "ReverseDiff"
        reverse_diff = ReverseDiff.gradient(f, at)
        forward = ForwardDiff.gradient(f, at)
        @test isapprox(reverse_diff, forward, rtol=rtol, atol=atol)
    end
end

_to_cov(B) = B * B' + Matrix(I, size(B)...)
