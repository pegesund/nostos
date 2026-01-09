-- List sum benchmark - 50M elements
-- Tests tail-recursive sum performance
-- Matches list_iterate.nos

{-# LANGUAGE BangPatterns #-}

-- Sum using tail recursion with strict accumulator
sumTR :: [Int] -> Int -> Int
sumTR [] !acc = acc
sumTR (x:xs) !acc = sumTR xs (acc + x)

main :: IO ()
main = do
    let list = [1..50000000]
    -- Print length to force list creation (prevents fusion)
    let !len = length list
    print len
    -- Sum the list
    let !result = sumTR list 0
    print result
