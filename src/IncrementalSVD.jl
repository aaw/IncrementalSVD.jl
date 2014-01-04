module IncrementalSVD

using ProgressMeter

export Rating, RatingSet, RatingsModel
export train
export split_ratings, rmse, truncate_model!, cosine_similarity
export items, users, item_features, user_features, show_items_by_features, item_search
export similar_items, similar_users, user_ratings, get_predicted_rating
export load_book_crossing_dataset, load_small_movielens_dataset, load_large_movielens_dataset

include("types.jl")
include("train.jl")
include("util.jl")

include("book_crossing_dataset.jl")
include("movielens_dataset.jl")

end
