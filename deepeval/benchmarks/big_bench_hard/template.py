from importlib import resources

from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask, bbh_confinement_statements_dict
from deepeval.benchmarks.big_bench_hard.cot_prompts import *
from deepeval.benchmarks.big_bench_hard.shot_prompts import *


class BigBenchHardTemplate:

    # COT prompts were taken directly from BBH Github Repo
    # Few-shot prompts were adpated from COT prompts by removing CoT Reasoning

    @staticmethod
    def generate_output(
        input: str, task: BigBenchHardTask, n_shots: int, enable_cot: bool, enable_analogy: bool
    ):
        folder = "cot_prompts" if enable_cot else "shot_prompts"
        filename = BigBenchHardTemplate.get_filename(task)

        # Construct the resource path
        package_path = f"deepeval.benchmarks.big_bench_hard.{folder}"

        # get prompt from text file based on n_shots and folder path
        prompt = "Task description: "
        prompt_content = BigBenchHardTemplate.read_file(package_path, filename)
        prompt += "\n\n".join(prompt_content[: (n_shots + 1) if not enable_analogy else 1])

        prompt += "\n\nQ: " + input + "\n"

        if enable_cot:
            prompt += "Let's think step-by-step."
        elif enable_analogy:
            prompt += f"Let's recall {n_shots if n_shots>0 else 1} relevant questions and answers. Finally, let's answer the initial question without explanations."

        prompt += f" {bbh_confinement_statements_dict[task]} Make sure to write the answer at the end."
        prompt += "\nA: "
        return prompt

    def read_file(package_path, filename):
        # Use resources.open_text to access the file within the package
        with resources.open_text(package_path, filename) as file:
            file_content = file.read()

        # Split the content into sections
        sections = file_content.split("\n\n")
        return sections

    def get_filename(task):
        # generate prompts
        return task.value + ".txt"
