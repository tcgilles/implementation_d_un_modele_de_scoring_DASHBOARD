import unittest
from main import (
    set_value_gauge, 
    update_credit_status, 
    plot_feature_importance_local,
    plot_feature_importance_global,
    plot_continuous_features,
    plot_box,
    plot_pie,
)

class Tests(unittest.TestCase):

    def test_set_value_gauge(self):
        result1 = set_value_gauge(100045)
        result2 = set_value_gauge(1)
        self.assertLessEqual(result1, 1)
        self.assertGreaterEqual(result1, 0)
        self.assertEqual(result2, 0)
    
    def test_update_credit_status(self):
        result1 = update_credit_status(100045, 0.1)
        result2 = update_credit_status(100020, 0.9)
        result3 = update_credit_status(None, 0.5)
        result4 = update_credit_status(1, 0.5)
        self.assertTupleEqual(
            result1,
            ("Crédit accordé", {"color": "green", "textAlign": "center"}) 
            )
        self.assertTupleEqual(
            result2,
            ("Crédit refusé", {"color": "red", "textAlign": "center"})
            )
        self.assertTupleEqual(result3, ("", {}))
        self.assertTupleEqual(
            result4,
            ("Identifiant incorrect", {"textAlign": "center", "font-size": 15})
            )

    def test_plot_feature_importance_local(self):
        result1 = plot_feature_importance_local(100045, 10)
        result2 = plot_feature_importance_local(1, 10)
        self.assertNotEqual(result1, {})
        self.assertEqual(result2, {})

    def test_plot_feature_importance_global(self):
        result1 = plot_feature_importance_global(100045, 10)
        result2 = plot_feature_importance_global(1, 10)
        self.assertNotEqual(result1, {})
        self.assertEqual(result2, {})

    def test_plot_continuous_features(self):
        result1 = plot_continuous_features(
            100045, "AMT_INCOME_TOTAL", "AMT_CREDIT", "Linear", "Log")
        result2 = plot_continuous_features(
            100045, "AMT_INCOME_TOTAL", None, "Linear", "Log")
        result3 = plot_continuous_features(
            100045, None, None, "Linear", "Log")
        result4 = plot_continuous_features(
            1, "AMT_INCOME_TOTAL", "AMT_CREDIT", "Linear", "Log")
        self.assertNotEqual(result1[0], {})
        self.assertNotEqual(result1[1], {})
        self.assertNotEqual(result1[2], {})
        self.assertNotEqual(result2[0], {})
        self.assertEqual(result2[1], {})
        self.assertEqual(result2[2], {})
        self.assertTupleEqual(result3, ({}, {}, {}))
        self.assertTupleEqual(result4, ({}, {}, {}))

    def test_plot_box(self):
        result1 = plot_box(100045, "CODE_GENDER")
        result2 = plot_box(100045, None)
        result3 = plot_box(1, "CODE_GENDER")
        self.assertNotEqual(result1, {})
        self.assertEqual(result2, {})
        self.assertEqual(result3, {})
    
    def test_plot_pie(self):
        result1 = plot_pie(100045, "CODE_GENDER")
        result2 = plot_pie(100045, None)
        result3 = plot_pie(1, "CODE_GENDER")
        self.assertNotEqual(result1, {})
        self.assertEqual(result2, {})
        self.assertEqual(result3, {})



    

    
