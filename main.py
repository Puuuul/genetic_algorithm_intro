import time
from collections import namedtuple
from functools import partial
from random import choices, randint, randrange, random
from typing import List, Tuple, Callable

Present = namedtuple("Present", ["name", "value", "price"])
Genome = List[int]
Population = List[Genome]
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]


# Prepare the wish list for further usage
def read_data(filename: str) -> List[Present]:
    with open(filename) as file:
        presents = []
        for line in file.readlines():
            line = line.split(":")
            name = line[0]
            value, price = [int(s) for s in (line[1].replace("$", "").strip().split(" "))]
            presents.append(Present(name, value, price))
        return presents


# This will surely work with more than 30 presents
def brute_force(presents: List[Present], max_price: int) -> int:
    best = (0, 0, "Nothing!")
    for n in range(2**len(presents)):
        genome = bin(n).replace("0b", "")[::-1]  # "Genome"
        price = 0
        value = 0
        for i, c in enumerate(genome):
            if int(c):
                price += presents[i].price
                value += presents[i].value
        if price > max_price:
            continue
        if value > best[0]:
            best = (value, price, genome)
    return best[0]


# Generate single genome
def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)


# Generate starting population
def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


# Calculate the fitness of a single genome
def fitness(genome: Genome, presents: List[Present], max_price: int) -> int:
    if sum([presents[i].price for i, g in enumerate(genome) if g]) > max_price:
        return 0
    else:
        return sum([presents[i].value for i, g in enumerate(genome) if g])


# "Reproduction"
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    p = randint(1, len(a) - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


# Mutate a genome
def mutation(genome: Genome, num: int = 2, probability: float = 0.5) -> Genome:
    for _ in range(num):
        i = randrange(len(genome))
        if random() > probability:
            genome[i] = 1 if genome[i] == 0 else 0
    return genome


# Select two parents to reproduce
def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(gene) for gene in population],
        k=2
    )


# Do the evolution
def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100) \
        -> Population:
    population = populate_func()

    for _ in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        next_generation = population[0:2]

        for _ in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population


if __name__ == "__main__":
    presents = read_data("list.txt")
    start = time.time()
    fitness_brute_force = brute_force(
        presents=presents,
        max_price=400
    )
    end = time.time()
    print(f"Brute Force took {end-start}s")
    print(f"Fitness value is: {fitness_brute_force}")
    start = time.time()
    population = run_evolution(
        populate_func=partial(generate_population, size=20, genome_length=len(presents)),
        fitness_func=partial(fitness, presents=presents, max_price=400)
    )
    end = time.time()
    print(f"Genetic Algorithm took {end - start}s")
    print(f"Fitness value is: {sum([presents[i].value for i, g in enumerate(population[0]) if int(g)])}")
