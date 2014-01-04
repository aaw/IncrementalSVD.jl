# Book Crossing Dataset (http://www.informatik.uni-freiburg.de/~cziegler/BX)
#
# Collected by Cai-Nicolas Ziegler in a 4-week crawl (August / September 2004) 
# from the Book-Crossing community with kind permission from Ron Hornbaker, CTO 
# of Humankind Systems. Contains 278,858 users (anonymized but with demographic 
# information) providing 1,149,780 ratings (explicit / implicit) about 271,379 
# books.
#
# Freely available for research use when acknowledged with the following 
# reference (further details on the dataset are given in this publication):
# Improving Recommendation Lists Through Topic Diversification,
# Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, Georg Lausen; 
# Proceedings of the 14th International World Wide Web Conference (WWW '05), 
# May 10-14, 2005, Chiba, Japan.
#
# Ratings are 0-10, but 0 means that the user didn't rate the book. There are
# also lots of users who have very few ratings and lots of books that have
# only been rated a few times, so these rating need a lot of cleaning before
# they're usuable. With the default parameters, we exclude any user who has
# rated less than 30 books and any book that's been rated less than 30 times,
# as well as any user who hasn't used at least 6 different ratings out of the
# range 0-10. Finally, we re-scale the ratings to the range 1-6 by mapping 0-5 
# to 1 and scaling all of the other ratings down by 4.

function load_book_crossing_dataset(;
                                    user_ratings_threshold=30, 
                                    book_ratings_threshold=30, 
                                    user_distinct_ratings_threshold=6,
                                    book_distinct_ratings_threshold=1)
    zipfile_name = "BX-CSV-Dump.zip"
    temp_dir = tempdir()
    download_path = joinpath(temp_dir, zipfile_name)
    if !isfile(download_path)
        println("Downloading book ratings data...")
        download("http://www.informatik.uni-freiburg.de/~cziegler/BX/$(zipfile_name)", download_path)
    else
        println("Reusing existing downloaded files...")
    end
    ratings_file_name = joinpath(temp_dir, "BX-Book-Ratings.csv")
    if !isfile(ratings_file_name)
        println("Extracting ratings data...")
        run(`unzip $download_path BX-Book-Ratings.csv -d $temp_dir`)
    end
    books_file_name = joinpath(temp_dir, "BX-Books.csv")
    if !isfile(books_file_name)
        println("Extracting books data...")
        run(`unzip $download_path BX-Books.csv -d $temp_dir`)
    end
    
    books_count = int(split(readall(`wc -l $(books_file_name)`), " ")[1]) - 1
    progress = Progress(books_count, 1, "Loading ISBN-to-book mapping ")
    book_file = open(books_file_name, "r")
    readline(book_file) # skip CSV header
    isbn_to_title = Dict{String,String}()
    for line in eachline(book_file)
        delim1 = search(line, "\";\"")
        delim2 = search(line, "\";\"", last(delim1))
        isbn, title = line[2:first(delim1)-1], line[last(delim1)+1:first(delim2)-1]
        isbn_to_title[isbn] = title
        next!(progress)
    end
    close(book_file)

    ratings_count = int(split(readall(`wc -l $(ratings_file_name)`), " ")[1]) - 1
    progress = Progress(ratings_count, 1, "Collecting counts of books and users for filtering ")
    user_ratings_count = Dict{String,Int32}()
    book_ratings_count = Dict{String,Int32}()
    user_distinct_ratings_set = Dict{String,Set{String}}()
    book_distinct_ratings_set = Dict{String,Set{String}}()
    ratings_file = open(ratings_file_name, "r")
    readline(ratings_file) # skip CSV header    
    for line in eachline(ratings_file)
        delim1 = search(line, "\";\"")
        delim2 = search(line, "\";\"", last(delim1))
        user, isbn, rating = line[2:first(delim1)-1], line[last(delim1)+1:first(delim2)-1], line[last(delim2)+1:end-3]
        user_ratings_count[user] = get(user_ratings_count, user, 0) + 1
        book_ratings_count[isbn] = get(book_ratings_count, isbn, 0) + 1
        if !haskey(user_distinct_ratings_set, user)
            user_distinct_ratings_set[user] = Set{String}()
        end
        push!(user_distinct_ratings_set[user], rating)
        if !haskey(book_distinct_ratings_set, isbn)
            book_distinct_ratings_set[isbn] = Set{String}()
        end
        push!(book_distinct_ratings_set[isbn], rating)
        next!(progress)
    end
    close(ratings_file)

    # Filter ratings while translating them into Rating structures.
    progress = Progress(ratings_count, 1, "Loading ratings ")
    ratings = (Rating)[]
    item_to_index = Dict{String, Int32}()
    user_to_index = Dict{String, Int32}()
    ratings_file = open(ratings_file_name, "r")
    readline(ratings_file) # skip CSV header
    for line in eachline(ratings_file)
        delim1 = search(line, "\";\"")
        delim2 = search(line, "\";\"", last(delim1))
        user, isbn, rating = line[2:first(delim1)-1], line[last(delim1)+1:first(delim2)-1], line[last(delim2)+1:end-3]
        if haskey(isbn_to_title, isbn) && # some isbns aren't in BX-Books.csv
           user_ratings_count[user] >= user_ratings_threshold && 
           book_ratings_count[isbn] >= book_ratings_threshold &&
           length(user_distinct_ratings_set[user]) >= user_distinct_ratings_threshold &&
           length(book_distinct_ratings_set[isbn]) >= book_distinct_ratings_threshold
            item = isbn_to_title[isbn]
            rating = max(float(rating) - 4, 1.0)
            user_index = get(user_to_index, user, 0)
            if user_index == 0
                user_index = user_to_index[user] = length(user_to_index) + 1
            end
            item_index = get(item_to_index, item, 0)
            if item_index == 0
                item_index = item_to_index[item] = length(item_to_index) + 1
            end
            push!(ratings, Rating(user_index, item_index, rating))
        end
        next!(progress)
    end
    close(ratings_file)
    training_set, test_set = split_ratings(ratings, 0.10)
    RatingSet(training_set, test_set, user_to_index, item_to_index)
end

