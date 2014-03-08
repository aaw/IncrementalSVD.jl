# Train a truncated SVD model of rank max_rank on the ratings in rating_set.
# min_epochs and max_epochs control the range of possible number of epochs used
# in the gradient descent. Since we train features sequentially, overfitting 
# can happen on early features. If min_epochs < max_epochs, a heuristic for 
# early exit is used that tends to avoid overfitting, tracking the rate of 
# change in the error between the model and the known ratings. This rate of
# change may decrease sharply initially, level off and increase, then finally
# decrease again. We try to catch it on it's second (final) decrease and
# terminate training the current feature when we do. learning_rate and 
# regularizer are the two standard gradient descent parameters that control
# how fast the descent happens and prevent the descent from over-aggressive
# training in any direction.

function train(rating_set::RatingSet,
               max_rank;
               min_epochs=0,
               max_epochs=100,
               learning_rate=0.001,
               regularizer=0.02)
    user_features = [0.1 for r=1:length(rating_set.user_to_index), c=1:max_rank]
    item_features = [0.1 for r=1:length(rating_set.item_to_index), c=1:max_rank]
    num_ratings = length(rating_set.training_set)
    residuals = [convert(Float32, 0.0) for r=1:num_ratings]
    p = Progress(max_epochs, 1, "Computing truncated rank $(max_rank) SVD ")
    for i=1:max_epochs
        for r=1:num_ratings
            residuals[r] = rating_set.training_set[r].value
        end
        for rank=1:max_rank
            for j=1:num_ratings
                rating = rating_set.training_set[j]
                item_feature = item_features[rating.item, rank]
                user_feature = user_features[rating.user, rank]
                residual = residuals[j] -= user_feature * item_feature
                item_features[rating.item, rank] += learning_rate * (residual * user_feature - regularizer * item_feature)
                user_features[rating.user, rank] += learning_rate * (residual * item_feature - regularizer * user_feature)
            end
        end
        next!(p)
    end

    # We end up with just the U and V matrices of the singular value 
    # decomposition, but S can be extracted by normalizing both U and V.
    singular_values = [norm(user_features[:,rank]) * norm(item_features[:,rank]) for rank=1:max_rank]
    for rank=1:max_rank
        user_features[:,rank] /= norm(user_features[:,rank])
        item_features[:,rank] /= norm(item_features[:,rank])
    end
    return RatingsModel(rating_set.user_to_index, 
                        rating_set.item_to_index, 
                        user_features, 
                        singular_values, 
                        item_features)
end
