-- List iteration benchmark - matches list_iterate.nos
-- Tests head/tail pattern matching performance

{-# LANGUAGE BangPatterns #-}

import Data.IORef
import Control.Exception (evaluate)
import Control.Monad (foldM)

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
sumTail [] !acc = acc
sumTail (x:xs) !acc = sumTail xs (acc + x)

-- Build test list
buildList :: Int -> [Int]
buildList 0 = []
buildList n = n : buildList (n - 1)

-- Force list spine and elements
forceList :: [Int] -> IO [Int]
forceList [] = return []
forceList (x:xs) = do
    _ <- evaluate x
    rest <- forceList xs
    return (x:rest)

-- Run function n times, accumulating results
loop :: Int -> Int -> (Int -> IO Int) -> IO Int
loop 0 !acc _ = return acc
loop n !acc f = do
    result <- f acc
    loop (n-1) result f

{-# NOINLINE runCount #-}
runCount :: [Int] -> Int
runCount list = countList list

{-# NOINLINE runSumTail #-}
runSumTail :: [Int] -> Int
runSumTail list = sumTail list 0

{-# NOINLINE runSumRec #-}
runSumRec :: [Int] -> Int
runSumRec list = sumRec list

main :: IO ()
main = do
    let n = 100000

    -- Build and force list - store in IORef to prevent sharing optimizations
    listRef <- newIORef =<< forceList (buildList n)

    -- Test 1: Count - 100 iterations (NOINLINE prevents fusion)
    c <- loop 100 0 $ \acc -> do
        list <- readIORef listRef
        return $! acc + runCount list
    print c

    -- Test 2: Sum tail recursive - 100 iterations
    s <- loop 100 0 $ \acc -> do
        list <- readIORef listRef
        return $! acc + runSumTail list
    print s

    -- Test 3: Sum non-tail recursive - 100 iterations
    r <- loop 100 0 $ \acc -> do
        list <- readIORef listRef
        return $! acc + runSumRec list
    print r

    print (c + s + r)
