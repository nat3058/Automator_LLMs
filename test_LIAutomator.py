import unittest
from together import Together
from meta_ai_api import MetaAI

from unittest.mock import Mock, patch

from vhil_OO import (
    LLMManager,
    MetaManager, 
    TogetherManager,
    VoiceAssistant,
    LIAutomator,
    ConnectionExecutor,
    PersonFinder,
    ScreenshotManager,
)

class TestLIAutomator(unittest.TestCase):
    def test_init(self):
        llm = Mock()
        vm = Mock()
        automator = LIAutomator(llm, vm)
        self.assertEqual(automator.llm, llm)
        self.assertEqual(automator.vm, vm)
        self.assertEqual(automator.autoskip_connection, False)

    def test_execute_full_connection_process(self):
        ss = ScreenshotManager(use_defaults=True)
        ss.config_ss_params()
        mm = MetaManager(new_llm_every_req=False) # False might be faster
        tm = TogetherManager(ss, new_llm_every_req=True) # new_llm_every_req=True necessary at this stage for TOGETHER 
        automator = LIAutomator(mm, tm, autoskip_connection=True)
        with patch.object(ConnectionExecutor, 'execute_connection') as mock_execute_connection:
            automator.execute_full_connection_process()
            mock_execute_connection.assert_called_once()

    # def test_execute_full_connection_process(self):
    #     llm = Mock()
    #     vm = Mock()
    #     automator = LIAutomator(llm, vm)
    #     with patch.object(automator, 'ask_vision_meta') as mock_ask_vision_meta:
    #         mock_ask_vision_meta.return_value = 'vision_res'
    #         with patch.object(ConnectionExecutor, 'execute_connection') as mock_execute_connection:
    #             automator.execute_full_connection_process()
    #             mock_execute_connection.assert_called_once()

    def test_execute_full_connection_process_with_error(self):
        llm = Mock()
        vm = Mock()
        automator = LIAutomator(llm, vm)
        with patch.object(automator, 'ask_vision_meta') as mock_ask_vision_meta:
            mock_ask_vision_meta.side_effect = Exception('Test exception')
            with self.assertRaises(Exception):
                automator.execute_full_connection_process()

    def test_find_next_person(self):
        llm = Mock()
        vm = Mock()
        automator = LIAutomator(llm, vm)
        with patch.object(PersonFinder, 'find_next_person') as mock_find_next_person:
            automator.find_next_person()
            mock_find_next_person.assert_called_once()

if __name__ == '__main__':
    unittest.main()