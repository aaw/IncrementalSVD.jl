# A single rating. The user and item are both represented by integer ids. A map
# between ids and user/item names is stored elsewhere, both in the rating set
# and the model.
immutable Rating
    user::Int32
    item::Int32
    value::Float32
end

# An internal type used only by the train function for caching between epochs.
immutable Residual
    value::Float32
    curr_error::Float32
    prev_error::Float32
end

# A set of ratings.
type RatingSet
    # The set of ratings that the model should be trained on.
    training_set::Array{Rating, 1}
    # The set of ratings that the model should be tested on. This is a hold-out
    # set, the model will never see these while being trained.
    test_set::Array{Rating, 1}
    # An index between human-readable user names and an interval of integer ids.
    user_to_index::Dict{AbstractString, Int32}
    # An index between human-readable item names and an interval of integer ids.
    item_to_index::Dict{AbstractString, Int32}
end

# An SVD model.
type RatingsModel
    # An index between human-readable user names and an interval of integer ids.
    user_to_index::Dict{AbstractString, Int32}
    # An index between human-readable item names and an interval of integer ids.
    item_to_index::Dict{AbstractString, Int32}
    # U, S, and V form the SVD decomposition of the original matrix, so
    # U * diagm(S) * V' will yield the model's approximation of the original
    # matrix. U is a (number of users) x (number of features) matrix, S is
    # a list of (number of features) singular values, and V is a (number of
    # items) x (number of features) matrix.
    U::AbstractArray{Float32,2}
    S::AbstractArray{Float32,1}
    V::AbstractArray{Float32,2}
end
