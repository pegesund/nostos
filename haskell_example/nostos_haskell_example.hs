{-# LANGUAGE BangPatterns #-}

import Control.Concurrent
import Control.Concurrent.Chan
import Control.Monad (replicateM_)

-- Worker function: sends its counter value to the channel
-- The ! before counter is a strictness annotation, ensuring counter is evaluated
-- before being put into the channel, which can sometimes help performance.
worker :: Chan Int -> Int -> IO ()
worker channel !counter = writeChan channel counter

-- Main function: spawns workers and sums their results
main :: IO ()
main = do
    let numWorkers = 100000 :: Int
    
    -- Create a new channel for workers to send their counter values
    channel <- newChan

    putStrLn $ "Spawning " ++ show numWorkers ++ " lightweight Haskell threads..."

    -- Spawn workers: Each worker gets a copy of the channel and its unique counter (0 to numWorkers-1)
    -- mapM_ takes a monadic action and applies it to each element of a list.
    -- forkIO is the monadic action that runs a computation in a new thread.
    mapM_ (\i -> forkIO $ worker channel i) [0 .. numWorkers - 1]

    putStrLn "All workers spawned! Collecting results..."

    -- Collect results from the channel and sum them
    -- Use a strict accumulator to avoid thunk buildup for large sums
    let loopSum !total !n =
            if n >= numWorkers
            then return total
            else do
                value <- readChan channel
                loopSum (total + value) (n + 1)
    
    finalSum <- loopSum 0 0

    putStrLn $ "All workers completed! Total sum of counters: " ++ show finalSum
