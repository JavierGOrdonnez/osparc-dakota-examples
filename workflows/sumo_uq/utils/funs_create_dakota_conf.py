### Useful functions to couple Python and Dakota - to use accross different scripts & notebooks
from typing import List, Optional
from pathlib import Path


def start_dakota_file(results_file_name="results.dat") -> str:
    """Make the start of a Dakota input file - making it produce a 'results.dat' file"""

    return f"""
    environment
        tabular_data
            tabular_data_file = '{results_file_name}'
    """


def add_adaptive_sampling(
    N_ADAPT: int,
    training_samples_file: Optional[str] = None,
    id_method: str = "ADAPTIVE_SAMPLING",
    model_pointer: str = "TRUE_MODEL",
    seed=43,
) -> str:
    s = f"""

    method
        id_method = '{id_method}'
        adaptive_sampling 
            max_iterations {N_ADAPT}
            samples_on_emulator = 1000 
            fitness_metric predicted_variance 
    """

    if training_samples_file:
        s += f"""
            import_build_points_file 
            "{training_samples_file}"
            {"custom_annotated header eval_id" if "_processed.txt" in training_samples_file else ""}  # only if processed file
            """
    else:
        raise ValueError("Training samples file must be provided for adaptive sampling")

    s += f"""
            model_pointer = "{model_pointer}"
            seed {seed}
            
            export_approx_points_file "predictions.dat"

            # response_levels # dont know how to use this
            # probability_levels # dont know how to use this
            # gen_reliability_levels # dont know how to use this

    """
    return s


def add_variables(
    variables: List[str],
    id_variables="VARIABLES",
    initial_points: Optional[List[float]] = None,
    lower_bounds: Optional[List[float]] = None,
    upper_bounds: Optional[List[float]] = None,
) -> str:
    vars_str = f"""

        variables
            continuous_design = {len(variables)}
            id_variables = '{id_variables}'
                descriptors {' '.join([f"'{v}'" for v in variables])}
        """
    if initial_points is not None:
        assert len(initial_points) == len(variables)
        vars_str += f"""
                initial_point     {' '.join([str(ip) for ip in initial_points])}
                """
    if lower_bounds is not None:
        assert len(lower_bounds) == len(variables)
        vars_str += f"""
                lower_bounds      {' '.join([str(lb) for lb in lower_bounds])}
                """
    if upper_bounds is not None:
        assert len(upper_bounds) == len(variables)
        vars_str += f"""
                upper_bounds      {' '.join([str(ub) for ub in upper_bounds])}
                """
    return vars_str


def add_interface_s4l(n_jobs: int = 2, id_model="S4L_MODEL") -> str:
    return f"""
        model
            id_model = '{id_model}'
            single
                interface_pointer = 'INTERFACE'
                responses_pointer = 'RESPONSES'
                variables_pointer = 'VARIABLES'

        interface,
            id_interface = 'INTERFACE'
            system asynchronous evaluation_concurrency = {n_jobs}
                analysis_drivers = "eval_s4l.cmd"
                    parameters_file = "x.in"
                    results_file    = "y.out"
        """


def add_responses(descriptors=["-AFpeak"]) -> str:
    descriptors = descriptors if isinstance(descriptors, list) else [descriptors]
    return f"""

        responses
            id_responses = 'RESPONSES'
            descriptors {f' '.join([f"'{d}'"  for d in descriptors])}
            objective_functions = {len(descriptors)}
            no_gradients
            no_hessians
        """


def add_surrogate_model(
    training_samples_file: str,
    id_model: str = "SURR_MODEL",
    surrogate_type: str = "gaussian_process surfpack",
    cross_validation_folds: Optional[int] = None,
) -> str:
    return f"""

        model
            id_model '{id_model}'
            surrogate global
                {surrogate_type}
                {"## hopefully faster by removing CV" if not cross_validation_folds 
                else f'''
                cross_validation folds = {cross_validation_folds} 
                metrics = "root_mean_squared" "sum_abs" "mean_abs" "max_abs" "rsquared"
                '''}
                import_build_points_file 
                    '{training_samples_file}'
                    custom_annotated header use_variable_labels eval_id 
                export_approx_points_file "predictions.dat"
                {'export_approx_variance_file "variances.dat"' if "gaussian_process" in surrogate_type else ""}
        """


def add_sampling_method(
    method: str = "lhs",
    model_pointer: Optional[str] = None,
    num_samples: int = 10,
    seed: int = 1234,
) -> str:
    return f"""
        method
            id_method = 'SAMPLING'
            sample_type
                {method}
            sampling
                samples = {num_samples}
                {f'seed = {seed}' if seed is not None else ""}
            {f'model_pointer = "{model_pointer}"' if model_pointer is not None else ""}
        """


def add_evaluation_method(
    input_file: str,
    model_pointer: str = "SURR_MODEL",
    includes_eval_id: bool = False,
) -> str:
    eval_str = f"""
        method
            id_method "EVALUATION"
            output debug
            model_pointer '{model_pointer}'
        """
    if input_file is not None:
        eval_str += f"""
            list_parameter_study
                import_points_file 
                    ## this file should be wo responses!!
                    '{input_file}'
                    custom_annotated header {'eval_id' if includes_eval_id else ''}
        """
    return eval_str


def add_moga_method(
    max_function_evaluations=1e6, max_iterations=1e6, population_size=64
):
    return f"""
        method
            moga
            output debug
            max_function_evaluations = {max_function_evaluations}
            max_iterations = {max_iterations}
            initialization_type unique_random
            niching_type radial 0.05 0.05
            population_size = {population_size}
        """


def add_evaluator_model(
    id_model="TRUE_MODEL",
    interface_pointer="INTERFACE",
    variables_pointer="VARIABLES",
    responses_pointer="RESPONSES",
):
    return f"""
        model
            id_model = '{id_model}'
            single
                interface_pointer = '{interface_pointer}'
                variables_pointer = '{variables_pointer}'
                responses_pointer = '{responses_pointer}'
        """


def add_python_interface(
    evaluation_function: str,
    id_interface="INTERFACE",
    batch_mode: bool = False,
):
    return f"""
        interface,
            id_interface = '{id_interface}' 
            {"batch" if batch_mode else ""}
            python 
                analysis_drivers
                    '{evaluation_function}'
            
        """


def write_to_file(dakota_conf_text, dakota_conf_path):
    dakota_conf_path = Path(dakota_conf_path)
    dakota_conf_path.write_text(dakota_conf_text)
    pass
