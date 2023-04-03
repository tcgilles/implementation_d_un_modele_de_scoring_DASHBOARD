import unittest
from dashboard import update_credit_status, set_value_gauge, set_options_feature_2
from dashboard import plot_dist_feature_2, plot_bivariate_graph

class Tests(unittest.TestCase):

    def test_credit_accepted(self):
        result = update_credit_status(100045)
        self.assertTupleEqual(result, ("Crédit accordé", {"color": "green", "textAlign": "center"}))

    def test_credit_declined(self):
        result = update_credit_status(100020)
        self.assertTupleEqual(result, ("Crédit refusé", {"color": "red", "textAlign": "center"}))

    def test_credit_invalid_id(self):
        result = update_credit_status(10)
        self.assertTupleEqual(result, ("Identifiant incorrect", {}))

    def test_value_gauge_incorrect_id(self):
        result = set_value_gauge(10)
        self.assertEqual(result, 1)

    def test_value_gauge_correct_id(self):
        result = set_value_gauge(100045)
        self.assertLessEqual(result, 1)
        self.assertGreaterEqual(result, 0)

    def test_empty_value_options_feature_2(self):
        result = set_options_feature_2("")
        self.assertTrue(result[1])

    def test_correct_value_options_feature_2(self):
        result = set_options_feature_2("AMT_CREDIT")
        self.assertFalse(result[1])
    
    def test_incorrect_input_dist_feature_2(self):
        result1 = plot_dist_feature_2(100045, "")
        result2 = plot_dist_feature_2(100, "AMT_CREDIT")
        self.assertEqual(len(result1), 0)
        self.assertEqual(len(result2), 0)
    
    def test_incorrect_input_bivariate_graph(self):
        result1 = plot_bivariate_graph(100045, "AMT_CREDIT", "", "Linear", "Linear")
        result2 = plot_bivariate_graph(100045, "", "AMT_INCOME_TOTAL", "Linear", "Linear")
        result3 = plot_bivariate_graph(100, "AMT_CREDIT", "AMT_INCOME_TOTAL", "Linear", "Linear")
        self.assertEqual(len(result1), 0)
        self.assertEqual(len(result2), 0)
        self.assertEqual(len(result3), 0)


    

    

    
