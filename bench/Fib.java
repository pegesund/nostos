// Fibonacci benchmark
public class Fib {
    static long fib(int n) {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }

    public static void main(String[] args) {
        long start = System.nanoTime();
        long result = fib(40);
        long elapsed = (System.nanoTime() - start) / 1_000_000; // convert to ms

        System.out.println("TIME:" + elapsed);
        System.out.println("RESULT:" + result);
    }
}
