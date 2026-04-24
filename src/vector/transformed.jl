for T in (B.UnivariateTransformed, B.MultivariateTransformed, B.MatrixTransformed)
    @eval begin
        to_linked_vec(td::$T) = to_linked_vec(td.dist) ∘ inverse(td.transform)
        from_linked_vec(td::$T) = td.transform ∘ from_linked_vec(td.dist)
        linked_vec_length(td::$T) = linked_vec_length(td.dist)
        linked_optic_vec(td::$T) = fill(nothing, linked_vec_length(td))
    end
end
