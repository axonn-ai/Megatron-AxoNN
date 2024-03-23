#credit - https://www.geeksforgeeks.org/java-program-for-closest-prime-number/

import bisect
import math
 
MAX = 100005
 
prime_numbers = []
 
# Sieve of Eratosthenes algorithm to find all prime numbers up to MAX
 
 
def sieve_of_eratosthenes():
    # Create a boolean array "prime[0..n]" and initialize all entries as true.
    # A value in prime[i] will finally be false if i is not a prime, else true.
    prime = [True] * (MAX + 1)
 
    # Update all multiples of p
    for p in range(2, int(math.sqrt(MAX)) + 1):
        # If prime[p] is not changed, then it is a prime
        if prime[p]:
            for i in range(p * p, MAX + 1, p):
                prime[i] = False
 
    # Add all prime numbers to the list
    for i in range(2, MAX + 1):
        if prime[i]:
            prime_numbers.append(i)
 
# Function to find the closest prime number to a given number
 
 
def closest_prime(number):
    # Handle special case of number 1 explicitly
    if number == 1:
        return 2
    else:
        # Generate all prime numbers using Sieve of Eratosthenes algorithm
        sieve_of_eratosthenes()
 
        # Find the index of the smallest element greater than number
        index = bisect.bisect_left(prime_numbers, number)
 
        # Check if the current element or the previous element is the closest
        if prime_numbers[index] == number or prime_numbers[index - 1] == number:
            return number
        elif abs(prime_numbers[index] - number) < abs(prime_numbers[index - 1] - number):
            return prime_numbers[index]
        else:
            return prime_numbers[index - 1]
