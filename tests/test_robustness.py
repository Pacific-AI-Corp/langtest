import unittest
from langtest.transform.robustness import *
from langtest.transform.constants import A2B_DICT
from langtest.utils.custom_types import SequenceClassificationSample
from langtest.utils.custom_types.sample import QASample, SummarizationSample
from langtest.transform import TestFactory


class RobustnessTestCase(unittest.TestCase):
    """
    Test case for the robustness of language transformations.
    """

    def setUp(self) -> None:
        """
        Set up test data.
        """
        self.sentences = [
            SequenceClassificationSample(
                original="I live in London, United Kingdom since 2019"
            ),
            SequenceClassificationSample(
                original="I cannot live in USA due to torandos caramelized"
            ),
        ]
        self.abbreviation_sentences = [
            SequenceClassificationSample(
                original="Please respond as soon as possible for the party tonight"
            ),
            SequenceClassificationSample(
                original="I cannot live in USA due to torandos caramelized"
            ),
        ]
        self.number_sentences = [
            SequenceClassificationSample(
                original="I live in London, United Kingdom since 2019"
            ),
            SequenceClassificationSample(
                original="I can't move to the USA because they have an average of 1000 tornadoes a year, and I'm terrified of them"
            ),
        ]
        self.sentences_with_punctuation = [
            SequenceClassificationSample(
                original="I live in London, United Kingdom since 2019."
            ),
            SequenceClassificationSample(
                original="I cannot live in USA due to torandos caramelized!"
            ),
        ]

        self.strip_all_punctuation = [
            SequenceClassificationSample(
                original="12 . dutasteride 0.5 mg Capsule Sig : One ( 1 ) Capsule PO once a day ."
            ),
            SequenceClassificationSample(
                original="In conclusion , RSDS is a relevant osteoarticular complication in patients receiving either anticalcineurinic drug ( CyA or tacrolimus ) , even under monotherapy or with a low steroid dose ."
            ),
        ]

        self.british_sentences = [
            SequenceClassificationSample(
                original="I live in London, United Kingdom since 2019"
            ),
            SequenceClassificationSample(
                original="I cannot live in USA due to torandos caramelised"
            ),
        ]
        self.contraction_sentences = [
            SequenceClassificationSample(
                original="I live in London, United Kingdom since 2019"
            ),
            SequenceClassificationSample(
                original="I can't live in USA due to torandos caramelized"
            ),
        ]
        self.gendered_sentences = [
            SequenceClassificationSample(original="He lives in the USA."),
            SequenceClassificationSample(
                original="He lives in the USA and his cat is black."
            ),
        ]
        self.dyslexia_sentences = [
            SequenceClassificationSample(original="I live in London of United Kingdom."),
            SequenceClassificationSample(original="I would like that."),
        ]
        self.ocr_sentences = [
            SequenceClassificationSample(
                original="organization's art can win tough acts."
            ),
            SequenceClassificationSample(
                original="Anyone can join our community garden."
            ),
        ]
        self.speech_to_text_sentences = [
            SequenceClassificationSample(
                original="I picked up a stone and attempted to skim it across the water."
            ),
            SequenceClassificationSample(
                original="This organization's art can win tough acts."
            ),
        ]
        self.add_slangify = [
            SequenceClassificationSample(
                original="I picked up a stone and attempted to skim it across the water."
            ),
            SequenceClassificationSample(
                original="It was totally excellent but useless bet."
            ),
        ]
        self.custom_proportion_lowercase = [
            SequenceClassificationSample(
                original="I PICKED UP A STONE AND ATTEMPTED TO SKIM IT ACROSS THE WATER."
            ),
            SequenceClassificationSample(
                original="IT WAS TOTALLY EXCELLENT BUT USELESS BET."
            ),
        ]
        self.custom_proportion_uppercase = [
            SequenceClassificationSample(
                original="i picked up a stone and attempted to skim it across the water."
            ),
            SequenceClassificationSample(
                original="it was totally excellent but useless bet."
            ),
        ]
        self.multipleperturbations = [
            SequenceClassificationSample(
                original="I live in London, United Kingdom since 2019"
            ),
            SequenceClassificationSample(
                original="I can't move to the USA because they have an average of 1000 tornadoes a year, and I'm terrified of them"
            ),
        ]
        self.adj_sentences = [
            SequenceClassificationSample(
                original="Lisa is wearing a beautiful shirt today. This soup is not edible."
            ),
            SequenceClassificationSample(original="They have a beautiful house."),
        ]
        self.age_sentences = [
            SequenceClassificationSample(original="I am 75 years old."),
            SequenceClassificationSample(original="The baby is 40 days old."),
        ]
        self.test_qa = [
            "20 euro note -- Until now there has been only one complete series of euro notes; however a new series, similar to the current one, is being released. The European Central Bank will, in due time, announce when banknotes from the first series lose legal tender status.",
            "is the first series 20 euro note still legal tender",
        ]
        self.add_contraction_QA_sample = [
            "Who is angry?",
            "We will be going to the beach today.",
        ]
        self.labels = [
            ["O", "O", "O", "B-LOC", "B-COUN", "I-COUN", "O", "B-DATE"],
            ["O", "O", "O", "O", "B-COUN", "O", "O", "O", "O", "O"],
        ]

        self.terminology = {
            "LOC": ["Chelsea"],
            "COUN": ["Spain", "Italy"],
            "DATE": ["2017"],
        }

    def test_uppercase(self) -> None:
        """
        Test the UpperCase transformation.
        """
        transformed_samples = UpperCase.transform(self.sentences)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(len(self.sentences), len(transformed_samples))
        for sample in transformed_samples:
            self.assertTrue(sample.test_case.isupper())

    def test_custom_proportion_uppercase(self) -> None:
        """
        Test the UpperCase transformation with custom proportion.
        """
        transformed_samples = UpperCase.transform(
            self.custom_proportion_uppercase, prob=0.6
        )
        self.assertIsInstance(transformed_samples, list)
        for sample in transformed_samples:
            self.assertNotEqual(sample.test_case, sample.original)

    def test_lowercase(self) -> None:
        """
        Test the LowerCase transformation.
        """
        transformed_samples = LowerCase.transform(self.sentences)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(len(self.sentences), len(transformed_samples))
        for sample in transformed_samples:
            self.assertTrue(sample.test_case.islower())

    def test_custom_proportion_lowercase(self) -> None:
        """
        Test the LowerCase transformation with custom proportion.
        """
        transformed_samples = LowerCase.transform(
            self.custom_proportion_lowercase, prob=0.6
        )
        self.assertIsInstance(transformed_samples, list)
        for sample in transformed_samples:
            self.assertNotEqual(sample.test_case, sample.original)

    def test_titlecase(self) -> None:
        """
        Test the TitleCase transformation.
        """
        transformed_samples = TitleCase.transform(self.sentences)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(len(self.sentences), len(transformed_samples))
        for sample in transformed_samples:
            self.assertTrue(sample.test_case.istitle())

    def test_add_punctuation(self) -> None:
        """
        Test the AddPunctuation transformation.
        """
        transformed_samples = AddPunctuation.transform(self.sentences)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(len(self.sentences), len(transformed_samples))
        for sample in transformed_samples:
            self.assertFalse(sample.test_case[-1].isalnum())
            self.assertEqual(len(sample.transformations), 1)

        transformed_samples = AddPunctuation.transform(self.sentences, count=2)
        self.assertEqual(len(transformed_samples), len(self.sentences) * 2)

    def test_strip_punctuation(self) -> None:
        """
        Test the StripPunctuation transformation.
        """
        transformed_samples = StripPunctuation.transform(self.sentences_with_punctuation)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(len(self.sentences), len(transformed_samples))
        for sample in transformed_samples:
            self.assertNotEqual(sample.test_case, sample.original)
            self.assertEqual(len(sample.transformations), 1)

    def test_strip_all_punctuation(self) -> None:
        """
        Test the StripAllPunctuation transformation.
        """
        transformed_samples = StripAllPunctuation.transform(self.strip_all_punctuation)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(len(self.sentences), len(transformed_samples))
        for sample in transformed_samples:
            self.assertNotEqual(sample.test_case, sample.original)
            self.assertEqual(len(sample.transformations), 3)

    def test_swap_entities(self) -> None:
        """
        Test the SwapEntities transformation.
        """
        transformed_samples = SwapEntities.transform(
            sample_list=self.sentences, labels=self.labels, terminology=self.terminology
        )
        for sample in transformed_samples:
            self.assertEqual(len(sample.transformations), 1)
            self.assertNotEqual(sample.test_case, sample.original)

        transformed_samples = SwapEntities.transform(
            sample_list=self.sentences,
            labels=self.labels,
            terminology=self.terminology,
            count=2,
        )
        self.assertEqual(len(transformed_samples), len(self.sentences) * 2)

    def test_american_to_british(self) -> None:
        """
        Test the ConvertAccent transformation.
        """
        transformed_samples = ConvertAccent.transform(
            sample_list=self.sentences, accent_map=A2B_DICT
        )
        self.assertIsInstance(transformed_samples, list)
        self.assertListEqual(
            [sample.test_case for sample in transformed_samples],
            [sample.original for sample in self.british_sentences],
        )

    def test_add_context(self) -> None:
        """
        Test the AddContext transformation.
        """
        start_context = ["Hello"]
        end_context = ["Bye"]
        transformed_samples = AddContext.transform(
            sample_list=self.sentences,
            starting_context=start_context,
            ending_context=end_context,
            strategy="combined",
        )

        self.assertIsInstance(transformed_samples, list)
        for sample in transformed_samples:
            self.assertTrue(sample.test_case.startswith(start_context[0]))
            self.assertTrue(sample.test_case.endswith(end_context[0]))
        transformed_samples = AddContext.transform(
            sample_list=self.sentences,
            starting_context=start_context,
            ending_context=end_context,
            strategy="combined",
            count=2,
        )
        self.assertEqual(len(transformed_samples), len(self.sentences) * 2)

    def test_add_contraction(self) -> None:
        """
        Test the AddContraction transformation
        """
        transformed_samples = AddContraction.transform(self.sentences)
        self.assertListEqual(
            [sample.test_case for sample in transformed_samples],
            [sample.original for sample in self.contraction_sentences],
        )
        self.assertEqual(
            [len(sample.transformations) for sample in transformed_samples], [0, 1]
        )

        expected_corrected_sentences = [
            "Who's angry?",
            "We'll be going to the beach today.",
        ]
        transformed_samples_qa = AddContraction.transform(self.add_contraction_QA_sample)
        self.assertIsInstance(transformed_samples, list)
        self.assertEqual(expected_corrected_sentences, transformed_samples_qa)

    def test_dyslexia_swap(self) -> None:
        """
        Test the DyslexiaWordSwap transformation.
        """
        transformed_samples = DyslexiaWordSwap.transform(self.dyslexia_sentences)
        self.assertIsInstance(transformed_samples, list)
        for sample in transformed_samples:
            self.assertTrue(sample.test_case != sample.original or sample.test_case)

    def test_number_to_word(self) -> None:
        """
        Test the NumberToWord transformation.
        """
        transformed_samples = NumberToWord.transform(self.number_sentences)
        self.assertIsInstance(transformed_samples, list)

    def test_add_ocr_typo(self) -> None:
        """
        Test the AddOcrTypo transformation.
        """
        expected_corrected_sentences = [
            "organization's a^rt c^an w^in tougb acts.",
            "anvone c^an j0in o^ur communitv gardcn.",
        ]
        transformed_samples = AddOcrTypo.transform(self.ocr_sentences)
        self.assertIsInstance(transformed_samples, list)
        self.assertListEqual(
            [sample.test_case for sample in transformed_samples],
            expected_corrected_sentences,
        )
        transformed_samples = AddOcrTypo.transform(self.ocr_sentences, count=2)
        self.assertEqual(len(transformed_samples), len(self.ocr_sentences) * 2)

    def test_abbreviation_insertion(self) -> None:
        """
        Test the AbbreviationInsertion transformation.
        """
        transformed_samples = AbbreviationInsertion.transform(self.abbreviation_sentences)
        self.assertIsInstance(transformed_samples, list)

    def test_add_speech_to_text_typo(self) -> None:
        """
        Test the AddSpeechToTextTypo transformation.
        """
        transformed_samples = AddSpeechToTextTypo.transform(self.speech_to_text_sentences)
        self.assertIsInstance(transformed_samples, list)
        for sample in transformed_samples:
            self.assertTrue(sample.test_case != sample.original or sample.test_case)
        transformed_samples = AddSpeechToTextTypo.transform(
            self.speech_to_text_sentences, count=2
        )
        self.assertEqual(len(transformed_samples), len(self.speech_to_text_sentences) * 2)

    def test_add_slangify_typo(self) -> None:
        """
        Test the AddSlangifyTypo transformation.
        """
        transformed_samples = AddSlangifyTypo.transform(self.add_slangify)
        self.assertIsInstance(transformed_samples, list)
        for sample in transformed_samples:
            self.assertNotEqual(sample.test_case, sample.original)

    def test_multipleperturbations(self) -> None:
        """
        Test the MultiplePerturbations transformation.
        """
        transformations = ["lowercase", "add_ocr_typo", "titlecase", "number_to_word"]

        transformed_samples = MultiplePerturbations.transform(
            self.multipleperturbations, transformations, config=None
        )
        self.assertIsInstance(transformed_samples, list)

        for sample in transformed_samples:
            self.assertNotEqual(sample.test_case, sample.original)

        original_qa = self.test_qa.copy()
        transformed_samples_qa = MultiplePerturbations.transform(
            self.test_qa, transformations, config=None
        )
        self.assertIsInstance(transformed_samples, list)
        self.assertNotEqual(original_qa, transformed_samples_qa)

    def test_adj_synonym_swap(self) -> None:
        """
        Test the AdjectiveSynonymSwap transformation.
        """
        transformed_samples = AdjectiveSynonymSwap.transform(self.adj_sentences)
        self.assertIsInstance(transformed_samples, list)
        for sample in transformed_samples:
            self.assertNotEqual(sample.test_case, sample.original)

    def test_adj_antonym_swap(self) -> None:
        """
        Test the AdjectiveSynonymSwap transformation.
        """
        transformed_samples = AdjectiveAntonymSwap.transform(self.adj_sentences)
        self.assertIsInstance(transformed_samples, list)
        for sample in transformed_samples:
            self.assertNotEqual(sample.test_case, sample.original)

    def test_random_age(self) -> None:
        """
        Test the RandomAge transformation.
        """
        transformed_samples = RandomAge.transform(self.age_sentences, random_amount=100)
        self.assertIsInstance(transformed_samples, list)
        for sample in transformed_samples:
            self.assertNotEqual(sample.test_case, sample.original)

    def test_randomize_options(self) -> None:
        """
        Test the RandomizeOptions transformation.
        """
        transformed_samples = RandomizeOptions.transform(
            sample_list=[
                QASample(
                    original_question="What is John Snow Labs?",
                    original_context="John Snow Labs is a healthcare company specializing in accelerating progress in data science.",
                    options="A. healthcare company\nB. tech company\nC. data science company",
                    task="question-answering",
                )
            ]
        )
        self.assertIsInstance(transformed_samples, list)
        for sample in transformed_samples:
            self.assertIsInstance(sample, ShuffleOptions)
            if isinstance(sample, ShuffleOptions):
                sample.shuffle()
                self.assertTrue(sample.options != sample.perturbed_options)


class RobustnessTestCaseQaAndSummarization(unittest.TestCase):
    """
    A test case class for testing QA and summarization samples on robustness classes.
    """

    def available_test(self) -> dict:
        """
        Get a dictionary of available robustness tests.

        Returns:
            dict: A dictionary containing available robustness tests.
        """
        tests = {
            j: i
            for i in BaseRobustness.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests

    def setUp(self) -> None:
        """
        Set up the test environment before each test.

        Returns:
            None
        """
        test_scenarios = TestFactory.test_scenarios()
        self.available_tests = {
            test: list(scenarios.keys()) for test, scenarios in test_scenarios.items()
        }

        self.perturbations_list = [
            i for i in self.available_tests["robustness"] if not i.startswith("image_")
        ]

        self.supported_tests = self.available_test()
        self.samples = {
            "question-answering": [
                QASample(
                    original_question="What is John Snow Labs?",
                    original_context="John Snow Labs is a healthcare company specializing in accelerating progress in data science.",
                    options="-",
                )
            ],
            "summarization": [
                SummarizationSample(
                    original="John Snow Labs is a healthcare company specializing in accelerating progress in data science.",
                )
            ],
        }

    def test(self) -> None:
        """
        Test QA and summarization sample for robustness classes.

        Returns:
            None
        """
        prob = 1.0
        for test in self.perturbations_list:
            for task in self.samples:
                sample = self.samples[task][-1]
                test_func = self.supported_tests[test].transform

                if test not in [
                    "swap_entities",
                    "american_to_british",
                    "british_to_american",
                    "add_context",
                    "multiple_perturbations",
                    "randomize_options",
                ]:
                    sample.transform(test_func, {}, prob)
                elif test in ["american_to_british", "british_to_american"]:
                    sample.transform(test_func, {"accent_map": A2B_DICT}, prob)
                elif test == "add_context":
                    sample.transform(
                        test_func,
                        {
                            "ending_context": ["Bye", "Reported"],
                            "starting_context": ["Hi", "Good morning", "hello"],
                        },
                        prob,
                    )
                elif test == "multiple_perturbations":
                    sample.transform(
                        test_func,
                        {},
                        prob,
                        perturbations=[
                            "lowercase",
                            "add_ocr_typo",
                            "titlecase",
                            "number_to_word",
                        ],
                    )

                if task == "question-answering":
                    assert (
                        sample.perturbed_question is not None
                        and sample.perturbed_context is not None
                    )
                else:
                    assert sample.test_case is not None
