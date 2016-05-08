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
    p = Progress(max_rank, 1, "Computing truncated rank $(max_rank) SVD ")
    training_set = convert(SharedArray, rating_set.training_set)
    user_features = SharedArray(Float32, (length(rating_set.user_to_index), max_rank), init = S -> S[Base.localindexes(S)] = 0.1)
    item_features = SharedArray(Float32, (length(rating_set.item_to_index), max_rank), init = S -> S[Base.localindexes(S)] = 0.1)
    residuals = SharedArray(Residual, size(rating_set.training_set), init = [Residual(rating.value, 0.0, 0.0) for rating in rating_set.training_set])
    num_ratings = length(rating_set.training_set)
    for rank=1:max_rank
      errors = SharedArray(Float32, (3,), init = S -> [0.0, Inf, Inf])
      for i=1:max_epochs
        @sync @parallel for j=1:num_ratings
          rating, residual = training_set[j], residuals[j]
          item_feature = item_features[rating.item, rank]
          user_feature = user_features[rating.user, rank]
          error_diff = residual.prev_error - residual.curr_error
          errors[1] += error_diff * error_diff
          residuals[j] = Residual(residual.value, -user_feature * item_feature + residual.value, residual.curr_error)
          item_features[rating.item, rank] += learning_rate * (residual.curr_error * user_feature - regularizer * item_feature)
          user_features[rating.user, rank] += learning_rate * (residual.curr_error * item_feature - regularizer * user_feature)
        end
        # distance decreases, then increases, then decreases. we want to catch it on second decrease
        if i > min_epochs && errors[1] < errors[2] && errors[2] > errors[3]
          break
        end
        errors[1], errors[2], errors[3] = 0.0, errors[1], errors[2]
      end
      for j=1:num_ratings
        residual = residuals[j]
        residuals[j] = Residual(residual.curr_error, residual.curr_error, 0)
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
