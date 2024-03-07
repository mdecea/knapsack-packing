import time
import numpy as np
from math import ceil
from multiprocessing import Pool

import KnapsackPacking.algorithms.evolutionary as evolutionary
import KnapsackPacking.algorithms.greedy as greedy
import KnapsackPacking.algorithms.reversible as reversible
from KnapsackPacking.algorithms.problem_solution import Item, Container, Problem
from KnapsackPacking.algorithms.common_algorithm_functions import get_time_since

algorithm_function = {
    "greedy": greedy.solve_problem,
    "reversible": reversible.solve_problem,
    "evolutionary": evolutionary.solve_problem,
}


def execute_algorithm_with_params(params):
    """Execute the algorithm specified in the first of the passed parameters
    with the rest of parameters, and return the solution, value and elapsed time"""

    # unpack the algorithm and its parameters
    (
        algorithm,
        algorithm_name,
        problem,
        show_solution_plot,
        solution_plot_save_path,
        calculate_times,
        calculate_value_evolution,
        rotation,
    ) = params

    start_time = time.time()
    value_evolution = None
    times_dict = None
    if calculate_value_evolution:
        if algorithm == evolutionary.solve_problem:
            if calculate_times:
                solution, times_dict, value_evolution = algorithm(
                    problem,
                    calculate_times=calculate_times,
                    return_population_fitness_per_generation=calculate_value_evolution,
                    rotation=rotation,
                )
            else:
                solution, value_evolution = algorithm(
                    problem,
                    calculate_times=calculate_times,
                    return_population_fitness_per_generation=calculate_value_evolution,
                    rotation=rotation,
                )
        else:
            if calculate_times:
                solution, times_dict, value_evolution = algorithm(
                    problem,
                    calculate_times=calculate_times,
                    return_value_evolution=calculate_value_evolution,
                    rotation=rotation,
                )
            else:
                solution, value_evolution = algorithm(
                    problem,
                    calculate_times=calculate_times,
                    return_value_evolution=calculate_value_evolution,
                    rotation=rotation,
                )
    elif calculate_times:
        solution, times_dict = algorithm(
            problem, calculate_times=calculate_times, rotation=rotation
        )
    else:
        solution = algorithm(problem, rotation=rotation)
    elapsed_time = get_time_since(start_time)

    if solution and (show_solution_plot or solution_plot_save_path):
        solution.visualize(
            show_plot=show_solution_plot, save_path=solution_plot_save_path
        )

    return solution, solution.value, value_evolution, elapsed_time, times_dict


def execute_algorithm(
    algorithm,
    algorithm_name,
    problem,
    show_solution_plot=False,
    solution_plot_save_path=None,
    calculate_times=False,
    calculate_fitness_stats=False,
    execution_num=1,
    process_num=1,
    rotation="none",
    stop_if_successful=True,
    optimal_value=None,
):
    """Execute the passed algorithm as many times as specified (with each execution in a different CPU process if indicated),
    returning (at least) lists with the obtained solutions, values and elapsed times (one per execution).add()

    If stop_if_successful = True, we will stop repetition when we have a solution that fits all the components"""

    # encapsulate the algorithm and its parameters in a tuple for each execution (needed for multi-processing)
    param_tuples = [
        (
            algorithm,
            algorithm_name,
            problem,
            show_solution_plot,
            solution_plot_save_path,
            calculate_times,
            calculate_fitness_stats,
            rotation,
        )
        for _ in range(execution_num)
    ]

    solutions, values, value_evolutions, times, time_divisions = (
        list(),
        list(),
        list(),
        list(),
        list(),
    )

    # if possible, perform each execution in a separate CPU process (in parallel)
    if process_num > 1:
        process_pool = Pool(process_num)
        batch_num = ceil(execution_num / process_num)
        for batch in range(batch_num):
            results = process_pool.map(
                execute_algorithm_with_params,
                param_tuples[batch * process_num : batch * process_num + process_num],
            )
            (
                batch_solutions,
                batch_values,
                batch_value_evolutions,
                batch_times,
                batch_time_divisions,
            ) = (
                [result[0] for result in results],
                [result[1] for result in results],
                [result[2] for result in results],
                [result[3] for result in results],
                [result[4] for result in results],
            )
            solutions.extend(batch_solutions)
            values.extend(batch_values)
            value_evolutions.extend(batch_value_evolutions)
            times.extend(batch_times)
            time_divisions.extend(batch_time_divisions)
            """process_pool.terminate()
            process_pool.join()"""

    # perform the calculation sequentially if multi-processing is not allowed
    else:
        for i in range(execution_num):
            print(f"Starting execution {i}")
            (
                solution,
                value,
                value_evolution,
                elapsed_time,
                time_division,
            ) = execute_algorithm_with_params(param_tuples[i])
            solutions.append(solution)
            values.append(value)
            value_evolutions.append(value_evolution)
            times.append(elapsed_time)
            time_divisions.append(time_division)

            if stop_if_successful and value == optimal_value:
                # Stop trying - we find a good solution
                return solutions, values, value_evolutions, times, time_divisions

    return solutions, values, value_evolutions, times, time_divisions


def solve_packing_problem(
    container_shape,
    element_shapes,
    weight_shapes=None,
    shape_rotation="none",
    algorithm="greedy",
    num_repeats=1,
    num_processes=1,
    plot_sol=False,
):
    """
    Solves the packing problem and returns the solution

    container: shape object with the container that has to fit all the shapes
    element_shapes: list of shape objects that have to be fitted in the container
    weight_shapes: if we want to prioritize some shapes to be fitted, we can
        provide a list of weights for each shape in element_shapes. Larger weights
        puts larger priority on them.
    shape_rotation: specifies if the shapes can be rotated or not
        "none": no rotation
        "free": any rotation
        "manhattan": only [0, 90, 180, 270] rotations allowed
    algorithm: which algorithm to use. "greedy", "evolutionary" or "reversible"
    num_repeats: number of times to repat the placement (these are optimization
        algorithms so no optimal solution is guaranteed).
    num_processes: number of processes to parallel run the multiple runs.
    """

    max_weight = np.Inf  # No maximum weight capacity of the container

    container = Container(max_weight, container_shape)

    if weight_shapes is None:
        weight_shapes = [1] * len(element_shapes)

    if len(weight_shapes) != len(element_shapes):
        raise ValueError(
            "The provided list of weights is not the same length as the list of shapes to fit."
        )

    items = []
    for item, weight in zip(element_shapes, weight_shapes):
        items.append(Item(item, weight, weight))

    problem = Problem(container, items)
    optimal_value = np.sum(weight_shapes)

    (
        solutions,
        _,
        _,
        _,
        _,
    ) = execute_algorithm(
        algorithm=algorithm_function[algorithm],
        algorithm_name=algorithm,
        problem=problem,
        execution_num=num_repeats,
        process_num=num_processes,
        calculate_times=False,
        calculate_fitness_stats=False,
        rotation=shape_rotation,
        optimal_value=optimal_value,
    )

    # Choose the best solution
    vals = []
    for sol in solutions:
        vals.append(
            np.sum(
                [
                    problem.items[item_index].weight
                    for item_index in sol.placed_items.keys()
                ]
            )
        )

    solution = solutions[np.argmax(vals)]

    if np.max(vals) != optimal_value:
        print("We could not fit all the components")

    # Plot solution if indicated
    if plot_sol:
        solution.visualize(
            title_override=f"{algorithm} solution",
            show_plot=True,
            save_path=None,
            show_item_value_and_weight=True,
            show_value_weight_ratio_bar=True,
        )

    return solution


if __name__ == "__main__":
    import KnapsackPacking.shapes.shape_functions as shape_functions

    container_shape = shape_functions.create_square((22.5, 22.5), 28)
    item_num = 100
    items = [shape_functions.create_square((0, 0), 4)] * item_num
    weight_shapes = None

    solution = solve_packing_problem(
        container_shape,
        items,
        weight_shapes=weight_shapes,
        algorithm="greedy",
        num_repeats=1,
        num_processes=1,
        plot_sol=True,
    )

    print(list(solution.placed_items.values())[0])
    print(type(solution.placed_items))
