# MovieLens dataset (http://grouplens.org/datasets/movielens)
#
# Please review the README files on the GroupLens site (link above) for the
# usage licenses and other details.
#
# The small dataset has 1 million ratings from 6,000 users over 4,000 movies.
# The large dataset has 10 million ratings from 72,000 users over 10,000 movies.
# Both datasets have ratings in the range 1.0-5.0.

function load_movielens_dataset(zipfile_name, archive_dir, split_ratio=0.10)
    temp_dir = tempdir()
    download_path = joinpath(temp_dir, zipfile_name)
    if !isfile(download_path)
        println("Downloading movie ratings data...")
        download("http://files.grouplens.org/datasets/movielens/$(zipfile_name)", download_path)
    else
        println("Reusing existing downloaded files...")
    end
    movie_file_name = joinpath(temp_dir, archive_dir, "movies.dat")
    ratings_file_name = joinpath(temp_dir, archive_dir, "ratings.dat")
    if !isfile(movie_file_name)
        println("Extracting movie data...")
        InfoZIP.unzip("$(download_path)", temp_dir)
    end
    original_id_to_movie = Dict{AbstractString, AbstractString}()
    movie_count = countlines(movie_file_name)
    progress = Progress(movie_count, 1, "Loading id-to-movie mapping ")
    open(movie_file_name, "r") do movie_file
        for line in eachline(movie_file)
            split_line = split(line, "::")
            original_id_to_movie[split_line[1]] = split_line[2]
            next!(progress)
        end
    end
    ratings = (Rating)[]
    item_to_index = Dict{AbstractString, Int32}()
    user_to_index = Dict{AbstractString, Int32}()
    ratings_count = countlines(ratings_file_name)
    progress = Progress(ratings_count, 1, "Loading ratings ")
    open(ratings_file_name, "r") do ratings_file
        for line in eachline(ratings_file)
            user, movie, rating, timestamp = split(line, "::")
            item = original_id_to_movie[movie]
            user_index = get(user_to_index, user, 0)
            if user_index == 0
                user_index = user_to_index[user] = length(user_to_index) + 1
            end
            item_index = get(item_to_index, item, 0)
            if item_index == 0
                item_index = item_to_index[item] = length(item_to_index) + 1
            end
            push!(ratings, Rating(user_index, item_index, parse.(Float32,rating)))
            next!(progress)
        end
    end
    training_set, test_set = split_ratings(ratings, split_ratio)
    RatingSet(training_set, test_set, user_to_index, item_to_index)
end

function load_small_movielens_dataset(split_ratio=0.10)
    load_movielens_dataset("ml-1m.zip", "ml-1m", split_ratio)
end

function load_large_movielens_dataset(split_ratio=0.10)
    load_movielens_dataset("ml-10m.zip", "ml-10M100K", split_ratio)
end
