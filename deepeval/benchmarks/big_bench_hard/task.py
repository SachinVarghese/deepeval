from enum import Enum

class BigBenchHardTask(Enum):
    BOOLEAN_EXPRESSIONS = "boolean_expressions"
    CAUSAL_JUDGEMENT = "causal_judgement"
    DATE_UNDERSTANDING = "date_understanding"
    DISAMBIGUATION_QA = "disambiguation_qa"
    DYCK_LANGUAGES = "dyck_languages"
    FORMAL_FALLACIES = "formal_fallacies"
    GEOMETRIC_SHAPES = "geometric_shapes"
    HYPERBATON = "hyperbaton"
    LOGICAL_DEDUCTION_FIVE_OBJECTS = "logical_deduction_five_objects"
    LOGICAL_DEDUCTION_SEVEN_OBJECTS = "logical_deduction_seven_objects"
    LOGICAL_DEDUCTION_THREE_OBJECTS = "logical_deduction_three_objects"
    MOVIE_RECOMMENDATION = "movie_recommendation"
    MULTISTEP_ARITHMETIC_TWO = "multistep_arithmetic_two"
    NAVIGATE = "navigate"
    OBJECT_COUNTING = "object_counting"
    PENGUINS_IN_A_TABLE = "penguins_in_a_table"
    REASONING_ABOUT_COLORED_OBJECTS = "reasoning_about_colored_objects"
    RUIN_NAMES = "ruin_names"
    SALIENT_TRANSLATION_ERROR_DETECTION = "salient_translation_error_detection"
    SNARKS = "snarks"
    SPORTS_UNDERSTANDING = "sports_understanding"
    TEMPORAL_SEQUENCES = "temporal_sequences"
    TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS = (
        "tracking_shuffled_objects_five_objects"
    )
    TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS = (
        "tracking_shuffled_objects_seven_objects"
    )
    TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS = (
        "tracking_shuffled_objects_three_objects"
    )
    WEB_OF_LIES = "web_of_lies"
    WORD_SORTING = "word_sorting"


bbh_confinement_statements_dict = {
    BigBenchHardTask.BOOLEAN_EXPRESSIONS: "\n\nOutput 'True' or 'False'. Full answer not needed.",
    BigBenchHardTask.CAUSAL_JUDGEMENT: "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    BigBenchHardTask.DATE_UNDERSTANDING: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', or '(F)'. Full answer not needed.",
    BigBenchHardTask.DISAMBIGUATION_QA: "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    BigBenchHardTask.DYCK_LANGUAGES: "\n\nOutput only the sequence of parenthases characters separated by white space. Full answer not needed.",
    BigBenchHardTask.FORMAL_FALLACIES: "\n\nOutput 'invalid' or 'valid'. Full answer not needed.",
    BigBenchHardTask.GEOMETRIC_SHAPES: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', or '(K)'. Full answer not needed.",
    BigBenchHardTask.HYPERBATON: "\n\nOutput '(A)' or'(B)'. Full answer not needed.",
    BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS: "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    BigBenchHardTask.LOGICAL_DEDUCTION_FIVE_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.LOGICAL_DEDUCTION_SEVEN_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', or '(G)'. Full answer not needed.",
    BigBenchHardTask.MOVIE_RECOMMENDATION: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.MULTISTEP_ARITHMETIC_TWO: "\n\nOutput the numerical answer. Full answer not needed.",
    BigBenchHardTask.NAVIGATE: "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    BigBenchHardTask.OBJECT_COUNTING: "\n\nOutput the numerical answer. Full answer not needed.",
    BigBenchHardTask.PENGUINS_IN_A_TABLE: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.REASONING_ABOUT_COLORED_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', or '(R)'. Full answer not needed.",
    BigBenchHardTask.RUIN_NAMES: "\n\nOutput '(A)', '(B)', '(C)', or '(D)'. Full answer not needed.",
    BigBenchHardTask.SALIENT_TRANSLATION_ERROR_DETECTION: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', or '(F)'. Full answer not needed.",
    BigBenchHardTask.SNARKS: "\n\nOutput '(A)' or'(B)'. Full answer not needed.",
    BigBenchHardTask.SPORTS_UNDERSTANDING: "\n\nOutput 'yes' or 'no'. Full answer not needed.",
    BigBenchHardTask.TEMPORAL_SEQUENCES: "\n\nOutput '(A)', '(B)', '(C)', or '(D)'. Full answer not needed.",
    BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS: "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', or '(G)'. Full answer not needed.",
    BigBenchHardTask.WEB_OF_LIES: "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    BigBenchHardTask.WORD_SORTING: "\n\nOutput only the sequence of words separated by white space. Full answer not needed.",
}
