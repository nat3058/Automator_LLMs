# Testing LLManager abstract class and child classes
import unittest
from together import Together
from meta_ai_api import MetaAI

from unittest.mock import Mock, patch

from vhil_OO import LLMManager, MetaManager, TogetherManager, VoiceAssistant

# class TestUser(unittest.TestCase):
#     def test_get_name(self):
#         user = User("John Doe", "john@example.com")
#         self.assertEqual(user.get_name(), "John Doe")

# if __name__ == "__main__":
#     unittest.main()
class TestMetaManager(unittest.TestCase):
    def test_init(self):
        manager = MetaManager()
        self.assertTrue(manager.new_llm_every_req)
        self.assertIsNone(manager.internalLLM)

        manager = MetaManager(new_llm_every_req=False)
        self.assertFalse(manager.new_llm_every_req)
        self.assertIsInstance(manager.internalLLM, MetaAI)

    def test_init_internal_llm(self):
        manager = MetaManager(new_llm_every_req=False)
        self.assertIsInstance(manager._init_internal_llm(), MetaAI)

    @patch.object(MetaAI, 'prompt')
    def test_prompt_llm_new_llm_every_req(self, mock_prompt):
        manager = MetaManager()
        prompt_str = 'Test prompt'
        response = manager.prompt_llm(prompt_str)
        mock_prompt.assert_called_once_with(prompt_str)
        self.assertEqual(response, mock_prompt.return_value)

    @patch.object(MetaAI, 'prompt')
    def test_prompt_llm_reuse_llm(self, mock_prompt):
        manager = MetaManager(new_llm_every_req=False)
        prompt_str = 'Test prompt'
        response = manager.prompt_llm(prompt_str)
        mock_prompt.assert_called_once_with(prompt_str)
        self.assertEqual(response, mock_prompt.return_value)

    def test_update_internal_llm(self):
        manager = MetaManager(new_llm_every_req=False)
        manager.total_numb_requests_served = MetaManager.MAX_REQS_B4_TIMEOUT
        self.assertIsInstance(manager.internalLLM, MetaAI)
        manager._update_internalLLM()
        self.assertNotEqual(manager.internalLLM, MetaAI())  # New instance should be created
    
    def test_init_internal_llm_called_once(self):
        with patch.object(MetaManager, '_init_internal_llm') as mock_init:
            manager = MetaManager(new_llm_every_req=False)
            mock_init.assert_called_once()

    def test_prompt_llm_new_llm_every_req_updates_total_requests(self):
        manager = MetaManager()
        manager.prompt_llm('Test prompt')
        self.assertEqual(manager.total_numb_requests_served, 1)

    def test_prompt_llm_reuse_llm_updates_total_requests(self):
        manager = MetaManager(new_llm_every_req=False)
        manager.prompt_llm('Test prompt')
        self.assertEqual(manager.total_numb_requests_served, 1)

    def test_update_internal_llm_called_when_max_requests_reached(self):
        manager = MetaManager(new_llm_every_req=False)
        manager.total_numb_requests_served = MetaManager.MAX_REQS_B4_TIMEOUT - 1
        with patch.object(MetaManager, '_update_internalLLM') as mock_init:
            manager.prompt_llm('Test prompt')
            mock_init.assert_called_once()

    def test_update_internal_llm_not_called_when_max_requests_not_reached(self):
        manager = MetaManager(new_llm_every_req=False)
        manager.total_numb_requests_served = MetaManager.MAX_REQS_B4_TIMEOUT - 2
        with patch.object(MetaManager, '_init_internal_llm') as mock_init:
            manager.prompt_llm('Test prompt')
            mock_init.assert_not_called()

class TestTogetherManager(unittest.TestCase):
    def test_init(self):
        ss_manager = Mock()
        manager = TogetherManager(ss_manager)
        self.assertEqual(manager.ss, ss_manager)
        self.assertEqual(manager.TOGETHER_model_str, "meta-llama/Llama-Vision-Free")

    def test_init_internal_llm(self):
        manager = TogetherManager(Mock())
        self.assertIsInstance(manager._init_internal_llm(), Together)

    def test_ask_vision_meta_success(self):
        ss_manager = Mock()
        ss_manager.take_ss.return_value = 'link_to_ss'
        manager = TogetherManager(ss_manager)
        with patch.object(manager, 'meta_vision_wrapper') as mock_meta_vision_wrapper:
            mock_meta_vision_wrapper.return_value = 'vision_res'
            res = manager.ask_vision_meta('q')
            self.assertEqual(res, 'vision_res')

    def test_ask_vision_meta_max_retries_exceeded(self):
        ss_manager = Mock()
        ss_manager.take_ss.return_value = 'link_to_ss'
        manager = TogetherManager(ss_manager)
        with patch.object(manager, 'meta_vision_wrapper') as mock_meta_vision_wrapper:
            mock_meta_vision_wrapper.side_effect = Exception('Test exception')
            res = manager.ask_vision_meta('q')
            self.assertIsNone(res)

    def test_ask_vision_meta_voice_assistant_called(self):
        ss_manager = Mock()
        ss_manager.take_ss.return_value = 'link_to_ss'
        manager = TogetherManager(ss_manager)
        with patch.object(VoiceAssistant, 'voice') as mock_voice:
            manager.ask_vision_meta('q')
            mock_voice.assert_called_once_with("I'm reading")

    # the link part is not right so TOGETHER API thinks its invalid req
    # def test_prompt_llm(self):
    #     ss_manager = Mock()
    #     ss_manager.take_ss.return_value = 'link_to_ss'
    #     manager = TogetherManager(ss_manager)
    #     with patch.object(manager, 'ask_vision_meta') as mock_ask_vision_meta:
    #         mock_ask_vision_meta.return_value = 'vision_res'
    #         res = manager.prompt_llm('prompt_str')
    #         self.assertEqual(res, 'vision_res')
    #         manager._increment_total_numb_requests_served.assert_called_once()

    #2nd batch of tests
    def test_TOGETHER_model_str_property(self):
        manager = TogetherManager(Mock())
        self.assertEqual(manager.TOGETHER_model_str, "meta-llama/Llama-Vision-Free")

    def test_ss_property(self):
        ss_manager = Mock()
        manager = TogetherManager(ss_manager)
        self.assertEqual(manager.ss, ss_manager)

    def test_ask_vision_meta_retries(self):
        ss_manager = Mock()
        ss_manager.take_ss.return_value = 'link_to_ss'
        manager = TogetherManager(ss_manager)
        with patch.object(manager, 'meta_vision_wrapper') as mock_meta_vision_wrapper:
            mock_meta_vision_wrapper.side_effect = [Exception('Test exception')] * 2 + ['vision_res']
            res = manager.ask_vision_meta('q')
            self.assertEqual(res, 'vision_res')

    def test_ask_vision_meta_timeout(self):
        ss_manager = Mock()
        ss_manager.take_ss.return_value = 'link_to_ss'
        manager = TogetherManager(ss_manager)
        with patch.object(manager, 'meta_vision_wrapper') as mock_meta_vision_wrapper:
            mock_meta_vision_wrapper.side_effect = [Exception('Test exception')] * TogetherManager.MAX_RETRIES
            res = manager.ask_vision_meta('q')
            self.assertIsNone(res)

    # the link part is not right so TOGETHER API thinks its invalid req            
    # def test_prompt_llm_retries(self):
    #     ss_manager = Mock()
    #     ss_manager.take_ss.return_value = 'link_to_ss'
    #     manager = TogetherManager(ss_manager)
    #     with patch.object(manager, 'ask_vision_meta') as mock_ask_vision_meta:
    #         mock_ask_vision_meta.side_effect = [Exception('Test exception')] * 2 + ['vision_res']
    #         res = manager.prompt_llm('prompt_str')
    #         self.assertEqual(res, 'vision_res')
    # def test_prompt_llm_timeout(self):
    #     ss_manager = Mock()
    #     ss_manager.take_ss.return_value = 'link_to_ss'
    #     manager = TogetherManager(ss_manager)
    #     with patch.object(manager, 'ask_vision_meta') as mock_ask_vision_meta:
    #         mock_ask_vision_meta.side_effect = [Exception('Test exception')] * TogetherManager.MAX_RETRIES
    #         res = manager.prompt_llm('prompt_str')
    #         self.assertIsNone(res)

if __name__ == '__main__':
    unittest.main()