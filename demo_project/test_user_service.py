import unittest

from user_service import create_user


class CreateUserTests(unittest.TestCase):
    def test_normalizes_user_fields(self):
        self.assertEqual(
            create_user({"email": " DEV@EXAMPLE.COM ", "name": " Dev "}),
            {"email": "dev@example.com", "name": "Dev"},
        )

    def test_rejects_invalid_payloads(self):
        for payload in (
            None,
            {"email": "not-an-email", "name": "Dev"},
            {"email": "dev@example.com", "name": " "},
            {"email": " ", "name": "Dev"},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                create_user(payload)


if __name__ == "__main__":
    unittest.main()
