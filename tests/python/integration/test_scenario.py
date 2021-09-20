# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario


TEST_PROJECT_KEY = "TESTAMAZONREKOGNITIONPLUGIN"


def test_run_object_detection(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="OBJECT_DETECTION")


def test_run_text_detection(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="TEXT_DETECTION")


def test_run_unsafe_content_detection(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="UNSAFE_CONTENT")
