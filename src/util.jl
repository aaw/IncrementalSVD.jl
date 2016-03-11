# Split a list of ratings into a training and test set, with at most
# target_percentage * length(ratings) in the test set. The property we want to
# preserve is: any user in some rating in the original set of ratings is also
# in the training set and any item in some rating in the original set of ratings
# is also in the training set. We preserve this property by iterating through
# the ratings in random order, only adding an item to the test set only if we
# haven't already hit target_percentage and we've already seen both the user
# and the item in some other ratings.
function split_ratings(ratings::Array{Rating,1},
                       target_percentage=0.10)
    seen_users = Set()
    seen_items = Set()
    training_set = (Rating)[]
    test_set = (Rating)[]
    shuffled = shuffle(ratings)
    for rating in shuffled
        if in(rating.user, seen_users) && in(rating.item, seen_items) && length(test_set) < target_percentage * length(shuffled)
            push!(test_set, rating)
        else
            push!(training_set, rating)
        end
        push!(seen_users, rating.user)
        push!(seen_items, rating.item)
    end
    return training_set, test_set
end

# RMSE measures the average difference between predicted and actual ratings.
# This function measures RMSE on a test subset of the rating set that isn't used
# for training the model. As an optional final parameter, you can pass in a rank
# that's less than the model's rank to see if you should have stopped training
# features earlier.
function rmse(rating_set::RatingSet, model::RatingsModel; rank=nothing)
    total = 0.0
    if rank == nothing
        rank = size(model.S,1)
    else
        rank = min(size(model.S,1), rank)
    end
    for rating in rating_set.test_set
        predicted = 0.0
        for i=1:rank
            predicted += model.U[rating.user,i] * model.S[i] * model.V[rating.item,i]
        end
        total += (predicted - rating.value) ^ 2
    end
    sqrt(total/length(rating_set.test_set))
end

# Reduce the rank of a model. If you have a model of rank 40 but find that
# rmse(rating_set, model, rank=30) < rmse(rating_set, model, rank=40), it may
# make sense to throw away the extra 10 dimensions by running truncate_model!(model, 30).
function truncate_model!(model::RatingsModel, rank)
    current_rank = size(model.U, 2)
    if current_rank > rank
        model.U = model.U[:,1:rank]
        model.V = model.V[:,1:rank]
    end
end

# Cosine similarity between two vectors x and y. The higher the cosine
# similarity, the smaller the angle between x and y.
function cosine_similarity(x,y)
    sum, norm1, norm2 = 0.0, 0.0, 0.0
    for (a,b) in zip(x,y)
        sum += a * b
        norm1 += a * a
        norm2 += b * b
    end
    return sum / (sqrt(norm1) * sqrt(norm2))
end

# All items that the model was trained with.
function items(model::RatingsModel)
    keys(model.item_to_index)
end

# All users that the model was trained with.
function users(model::RatingsModel)
    keys(model.user_to_index)
end

# Extract the feature vector for an item.
function item_features(model::RatingsModel, item)
    model.V[model.item_to_index[item],:]
end

# Extract the feature vector for a user.
function user_features(model::RatingsModel, user)
    model.U[model.user_to_index[user],:]
end

# Returns a list of items, sorted by their values for the kth feature. This can
# help you interpret what the kth feature actually means.
function show_items_by_feature(model::RatingsModel, k)
    sort([i for i in items(model)], by=(m -> model.V[model.item_to_index[m],k]))
end

# Search for a particular item by a case-insensitive substring.
function item_search(model::RatingsModel, text)
    results = (AbstractString)[]
    r = Regex(".*?$(lowercase(text)).*")
    for item in items(model)
        if ismatch(r, lowercase(item))
            push!(results, item)
        end
    end
    return results
end

# Return the most similar items to a given item based on cosine similarity.
function similar_items(model::RatingsModel, item; max_results=10)
    features = item_features(model, item)
    sort([i for i in items(model)], by=(m -> -cosine_similarity(item_features(model, m), features)))[1:max_results]
end

# Return the most similar users to a given user based on cosine similarity.
function similar_users(model::RatingsModel, user; max_results=10)
    features = user_features(model, user)
    sort([i for i in users(model)], by=(u -> -cosine_similarity(user_features(model, u), features)))[1:max_results]
end

# Return all of the ratings in the training set for a given user as (item, rating) tuples.
function user_ratings(rating_set::RatingSet, user)
    user_id = rating_set.user_to_index[user]
    id_to_item = [id => item for (item, id) in rating_set.item_to_index]
    extract_rating = rating -> (id_to_item[rating.item], rating.value)
    is_user = rating -> rating.user == user_id
    map(extract_rating, sort(filter(is_user, rating_set.training_set), by=r -> -r.value))
end

# Return the the model's predicted rating for a given user and item.
function get_predicted_rating(model::RatingsModel, user, item)
    sum([model.U[model.user_to_index[user],r] * model.S[r] * model.V[model.item_to_index[item],r] for r=1:size(model.S,1)])
end
