
import unittest
import os
import uuid
from unittest.mock import patch, MagicMock

# Set environment variables for testing
os.environ['SUPABASE_URL'] = 'http://localhost:54321'
os.environ['SUPABASE_KEY'] = 'test_key'
os.environ['SUPABASE_SERVICE_ROLE_KEY'] = 'test_service_role_key'
os.environ['REDIS_URL'] = 'redis://localhost:6379/0'


class UsageLimitsTestCase(unittest.TestCase):

    def setUp(self):
        with patch.dict(os.environ, {"SUPABASE_URL": "http://localhost:54321", "SUPABASE_SERVICE_ROLE_KEY": "test_service_role_key"}, clear=True):
            from app import app, db_utils, get_user_status
            from utils.subscription_utils import PLANS
            self.app = app.test_client()
            self.app.testing = True
            self.db_utils = db_utils
            self.get_user_status = get_user_status
            self.PLANS = PLANS

    @patch('utils.db_utils.get_supabase_admin_client')
    def test_non_whop_user_limits(self, mock_get_supabase_admin_client):
        """
        Test case for a non-whop user.
        They should be on the 'free' plan and their queries should be counted.
        """
        user_id = str(uuid.uuid4())

        # Mock the profile and usage stats from the database
        mock_supabase_client = MagicMock()
        mock_get_supabase_admin_client.return_value = mock_supabase_client
        mock_supabase_client.table().select().eq().maybe_single().execute.return_value.data = {
            'id': user_id,
            'whop_user_id': None,
            'personal_plan_id': None,
            'queries_this_month': 0,
            'channels_processed': 0
        }

        # 1. Check if the user is assigned the 'free' plan
        with app.app_context():
            user_status = get_user_status(user_id)
            self.assertEqual(user_status['plan_id'], 'free')
            self.assertEqual(user_status['limits']['max_channels'], 2)
            self.assertEqual(user_status['limits']['max_queries_per_month'], 50)

        # 2. Simulate asking a question and check if query count is incremented
        with app.app_context():
            # Mock the RPC call
            mock_rpc = MagicMock()
            mock_supabase_client.rpc = mock_rpc

            # This is a simplified call to the function that would be in the callback
            db_utils.increment_personal_query_usage(user_id)

            # Assert that the correct RPC function was called
            mock_rpc.assert_called_once_with('increment_personal_query_usage', {'p_user_id': user_id})

    @patch('utils.db_utils.get_supabase_admin_client')
    def test_whop_community_member_limits(self, mock_get_supabase_admin_client):
        """
        Test case for a whop community member without a personal plan.
        They should be on the 'community_member' plan, unable to add personal channels.
        """
        user_id = str(uuid.uuid4())
        community_id = str(uuid.uuid4())

        # Mock the profile and usage stats from the database
        mock_supabase_client = MagicMock()
        mock_get_supabase_admin_client.return_value = mock_supabase_client
        mock_supabase_client.table().select().eq().maybe_single().execute.return_value.data = {
            'id': user_id,
            'whop_user_id': 'whop_user_123',
            'personal_plan_id': None,
            'queries_this_month': 0,
            'channels_processed': 0
        }

        # 1. Check if the user is assigned the 'community_member' plan
        with app.app_context():
            user_status = get_user_status(user_id, active_community_id=community_id)
            self.assertEqual(user_status['plan_id'], 'community_member')
            self.assertEqual(user_status['limits']['max_channels'], 0) # Key check for this user type

    @patch('utils.db_utils.get_supabase_admin_client')
    def test_whop_upgraded_member_limits(self, mock_get_supabase_admin_client):
        """
        Test case for a whop community member who has upgraded their personal plan.
        """
        user_id = str(uuid.uuid4())
        community_id = str(uuid.uuid4())

        # Mock the profile and usage stats from the database
        mock_supabase_client = MagicMock()
        mock_get_supabase_admin_client.return_value = mock_supabase_client
        mock_supabase_client.table().select().eq().maybe_single().execute.return_value.data = {
            'id': user_id,
            'whop_user_id': 'whop_user_456',
            'personal_plan_id': 'whop_pro_member', # Upgraded plan
            'queries_this_month': 0,
            'channels_processed': 0
        }

        # 1. Check if the user is assigned the correct upgraded plan
        with app.app_context():
            user_status = get_user_status(user_id, active_community_id=community_id)
            self.assertEqual(user_status['plan_id'], 'whop_pro_member')
            self.assertEqual(user_status['limits']['max_channels'], PLANS['whop_pro_member']['max_channels'])


if __name__ == '__main__':
    unittest.main()
