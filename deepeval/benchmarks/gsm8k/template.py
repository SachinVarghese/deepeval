import re


class GSM8KTemplate:

    # Template was inspired by https://arxiv.org/pdf/2110.14168.pdf
    # Original method trained the generator on training set
    # Here we use the training set for COT few_shot prompting

    @staticmethod
    def generate_output(
        input: str, train_set: object, n_shots: int, enable_cot: bool, enable_analogy: bool
    ):
        prompt = ""

        # generate examples for n_shot inference
        if not enable_analogy:
            if n_shots > 0:
                prompt = "The following are grade school math word problems\n\n"
            for i in range(n_shots):
                prompt += (
                    GSM8KTemplate.format_example(train_set[i], enable_cot) + "\n\n"
                )

        # problem of interest
        prompt += "**Problem**: " + input
        prompt += "\n**Answer**: "

        if enable_cot:
            prompt += "Let's think step-by-step."
        elif enable_analogy:
            prompt += f"Let's recall {n_shots if n_shots>0 else 1} relevant problems and answers. Finally, let's answer the initial problem without explanations."
        else:
            prompt += "No explanation needed."
        prompt += " Make sure to display only the numerical answer at the end.\n"
        return prompt

    @staticmethod
    def format_example(data: dict, enable_cot: bool):

        formatted_problem = ""
        question = data["question"]
        formatted_problem += "**Problem**: " + question + "\n"

        raw_answer = data["answer"]
        solution, answer = raw_answer.strip().split("\n#### ")
        if enable_cot:
            formatted_problem += "**Solution**: " + solution + "\n"
        formatted_problem += "**Answer**: " + answer

        return formatted_problem

    @staticmethod
    def format_answer(data: dict):
        raw_answer = data["answer"]
        answer = re.findall(r"#### (.*)", raw_answer)[0]
        return answer

    def format_subject(subject: str):
        return
