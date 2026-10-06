import unittest

from user_service import create_user


class CreateUserTests(unittest.TestCase):
    def test_returns_user_fields(self):
        self.assertEqual(
            create_user({"email": "dev@example.com", "name": "Dev"}),
            {"email": "dev@example.com", "name": "Dev"},
        )


if __name__ == "__main__":
    unittest.main()