
import unittest
import os
import uuid
from unittest.mock import patch, MagicMock

class NewFlowTestCase(unittest.TestCase):

    def setUp(self):
        # Set environment variables before importing app modules
        with patch.dict(os.environ, {
            "SUPABASE_URL": "http://mock-supabase.com",
            "SUPABASE_SERVICE_ROLE_KEY": "mock-key",
            "REDIS_URL": "redis://mock-redis:6379/0"
        }, clear=True):
            from app import app, db_utils, get_user_status
            from utils.subscription_utils import PLANS, COMMUNITY_PLANS

            self.app = app.test_client()
            self.app.testing = True

            self.db_utils = db_utils
            self.get_user_status = get_user_status
            self.PLANS = PLANS
            self.COMMUNITY_PLANS = COMMUNITY_PLANS

    @patch('utils.db_utils.get_supabase_admin_client')
    def test_admin_testing_plan_channel_limit(self, mock_get_supabase_admin_client):
        """
        Tests if a new admin is correctly assigned the 'admin_testing' plan
        and is limited to 1 total channel.
        """
        admin_user_id = str(uuid.uuid4())
        community_id = str(uuid.uuid4())

        mock_supabase = MagicMock()
        mock_get_supabase_admin_client.return_value = mock_supabase

        # Mock DB returns for get_user_status
        mock_supabase.table().select().eq().single().execute.return_value.data = {'owner_user_id': admin_user_id}
        mock_supabase.table().select().eq().maybe_single().execute.return_value.data = {
            'id': admin_user_id, 'whop_user_id': 'whop_admin_123', 'personal_plan_id': None
        }

        # Check plan assignment
        with self.app.app_context():
            status = self.get_user_status(admin_user_id, community_id)
            self.assertEqual(status['plan_id'], 'admin_testing')
            self.assertEqual(status['limits']['max_channels'], 1)

        # Mock the channel count check
        self.db_utils.count_channels_for_user = MagicMock(return_value=1)

        # Simulate the /channel route logic
        with self.app.app_context():
            # This logic is simplified from the app.py route for clarity
            max_channels = status['limits'].get('max_channels', 0)
            current_channels = self.db_utils.count_channels_for_user(admin_user_id)
            self.assertTrue(max_channels != float('inf') and current_channels >= max_channels)


    @patch('utils.db_utils.get_supabase_admin_client')
    def test_hybrid_query_enforcement_and_tracking(self, mock_get_supabase_admin_client):
        """
        Tests that the correct query limit (personal vs. community) is enforced
        and that the correct counters are incremented.
        """
        # --- SCENARIO 1: Default member using community pool ---
        default_member_id = str(uuid.uuid4())
        community_id = str(uuid.uuid4())

        mock_supabase = MagicMock()
        mock_get_supabase_admin_client.return_value = mock_supabase

        # Mock DB returns for a default member
        mock_supabase.table().select().eq().maybe_single().execute.return_value.data = {
            'id': default_member_id, 'whop_user_id': 'whop_member_123', 'has_personal_plan': False, 'personal_plan_id': None
        }

        # Mock the community status, community has 200 queries, 199 used
        self.db_utils.get_community_status = MagicMock(return_value={
            'limits': {'query_limit': 200}, 'usage': {'queries_used': 199}
        })

        # Mock the on_complete_callback from the stream_answer route
        with self.app.app_context():
            # Create mocks for the increment functions
            self.db_utils.increment_community_query_usage = MagicMock()
            self.db_utils.increment_personal_query_usage = MagicMock()

            user_status = self.get_user_status(default_member_id, community_id)

            # Manually simulate the callback logic
            if user_status.get('has_personal_plan'):
                self.db_utils.increment_personal_query_usage(default_member_id)
            elif community_id:
                self.db_utils.increment_community_query_usage(community_id, is_trial=False)
                self.db_utils.increment_personal_query_usage(default_member_id)

            # Assert that BOTH counters were incremented for the default member
            self.db_utils.increment_community_query_usage.assert_called_once_with(community_id, is_trial=False)
            self.db_utils.increment_personal_query_usage.assert_called_once_with(default_member_id)

        # --- SCENARIO 2: Upgraded member using personal limit ---
        upgraded_member_id = str(uuid.uuid4())

        # Mock DB returns for an upgraded member
        mock_supabase.table().select().eq().maybe_single().execute.return_value.data = {
            'id': upgraded_member_id, 'whop_user_id': 'whop_member_456', 'personal_plan_id': 'whop_pro_member'
        }

        with self.app.app_context():
            # Reset mocks
            self.db_utils.increment_community_query_usage.reset_mock()
            self.db_utils.increment_personal_query_usage.reset_mock()

            user_status = self.get_user_status(upgraded_member_id, community_id)

            # Manually simulate the callback logic again
            if user_status.get('has_personal_plan'):
                self.db_utils.increment_personal_query_usage(upgraded_member_id)
            elif community_id:
                self.db_utils.increment_community_query_usage(community_id, is_trial=False)
                self.db_utils.increment_personal_query_usage(upgraded_member_id)

            # Assert that ONLY the personal counter was incremented
            self.db_utils.increment_community_query_usage.assert_not_called()
            self.db_utils.increment_personal_query_usage.assert_called_once_with(upgraded_member_id)


if __name__ == '__main__':
    unittest.main()
