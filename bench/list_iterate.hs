-- List iteration benchmark - matches list_iterate.nos
-- Tests head/tail pattern matching performance

-- Count elements using pattern matching
countList :: [a] -> Int
countList [] = 0
countList (_:xs) = 1 + countList xs

-- Sum using pattern matching (not tail recursive)
sumRec :: [Int] -> Int
sumRec [] = 0
sumRec (x:xs) = x + sumRec xs

-- Sum using tail recursion with pattern matching
sumTail :: [Int] -> Int -> Int
sumTail [] acc = acc
sumTail (x:xs) acc = sumTail xs (acc + x)

-- Find max using pattern matching
maxRec :: [Int] -> Int
maxRec [x] = x
maxRec (x:xs) = let m = maxRec xs in if x > m then x else m

-- Reverse using pattern matching (builds new list)
reverseAcc :: [a] -> [a] -> [a]
reverseAcc [] acc = acc
reverseAcc (x:xs) acc = reverseAcc xs (x:acc)

-- Build test list
buildList :: Int -> [Int]
buildList 0 = []
buildList n = n : buildList (n - 1)

main :: IO ()
main = do
    let n = 50000

    -- Build list once
    let list = buildList n

    -- Test 1: Count (pure iteration)
    let c1 = countList list
    let c2 = countList list
    let c3 = countList list

    -- Test 2: Sum tail recursive (pure iteration)
    let s1 = sumTail list 0
    let s2 = sumTail list 0
    let s3 = sumTail list 0

    -- Test 3: Sum non-tail recursive
    let r1 = sumRec list
    let r2 = sumRec list
    let r3 = sumRec list

    -- Force evaluation and print
    print c1
    print s1
    print r1

    print (c1 + s1 + r1)
