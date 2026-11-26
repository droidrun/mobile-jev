import unittest

from mobilerun_agents.agent.action_selector.llm.parsing import (
    ToolCallParseStatus,
    extract_add_memory,
    parse_tool_calls_detailed,
)


class SelectorXmlParserTest(unittest.TestCase):
    def test_classifies_plain_text_as_no_markup(self):
        result = parse_tool_calls_detailed("I need to inspect the current screen.")

        self.assertEqual(result.status, ToolCallParseStatus.NO_MARKUP)
        self.assertEqual(result.thought, "I need to inspect the current screen.")
        self.assertEqual(result.calls, [])

    def test_classifies_discord_dsml_fixture_as_malformed(self):
        text = """I'm still on the main feed. I need to tap on the Profile tab.

<function_calls>
<invoke name="click">
<｜DSML｜ name="index">189</｜DSML｜>
</invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertIn("Profile tab", result.thought)
        self.assertEqual(result.calls, [])

    def test_classifies_captured_hybrid_dsml_as_malformed(self):
        text = """I will return to Settings home.
<function_calls>
<invoke name="system_button">
<｜｜DSML｜｜rameter name="button">back</｜｜DSML｜｜rameter>
</｜｜DSML｜｜invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_rejects_dsml_corruption_inside_parameter_value(self):
        text = """Settings home is visible. Marking the task complete.
<function_calls>
<invoke name="complete">
<parameter name="success">true</｜｜DSML｜｜>
<parameter name="message">Completed 20 verified cycles.</parameter>
</invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_rejects_dsml_corruption_crossing_xml_parameter_variant(self):
        text = """Settings home is visible. Marking the task complete.
<function_calls>
<invoke name="complete">
<parameter name="success">true</｜｜DSML｜｜>
<parameter name = "message">Completed 20 verified cycles.</parameter>
</invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_allows_ordinary_dsml_text_inside_parameter_value(self):
        text = """I will type the literal markup.
<function_calls>
<invoke name="type_text">
<parameter name="text"><span title="DSML">literal payload</span></parameter>
</invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.VALID)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            result.calls[0].parameters,
            {"text": '<span title="DSML">literal payload</span>'},
        )

    def test_allows_literal_dsml_syntax_inside_parameter_value(self):
        text = """I will type the literal DSML syntax.
<function_calls>
<invoke name="type_text">
<parameter name="text">before <｜DSML｜tool_calls>literal</｜DSML｜tool_calls> after</parameter>
</invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.VALID)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            result.calls[0].parameters,
            {"text": "before <｜DSML｜tool_calls>literal</｜DSML｜tool_calls> after"},
        )

    def test_literal_dsml_payload_does_not_hide_structural_dsml(self):
        text = """I will type text and then tap.
<function_calls>
<invoke name="type_text">
<parameter name="text">literal <｜DSML｜payload></parameter>
</invoke>
<｜DSML｜ name="click">
<parameter name="index">12</parameter>
</｜DSML｜>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_allows_unmatched_tool_like_text_inside_parameter_value(self):
        text = """I will type the literal snippet.
<function_calls>
<invoke name="type_text">
<parameter name="text">literal <invoke name="example"> token</parameter>
</invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.VALID)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            result.calls[0].parameters,
            {"text": 'literal <invoke name="example"> token'},
        )

    def test_classifies_dsml_without_xml_wrapper_as_malformed(self):
        text = """I will tap the target.
<｜DSML｜tool_calls>
<｜DSML｜invoke name="click">
<｜DSML｜parameter name="index">12</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.thought, "I will tap the target.")
        self.assertEqual(result.calls, [])

    def test_classifies_standalone_invoke_as_malformed(self):
        text = """I will tap the target.
<invoke name="click"><parameter name="index">12</parameter></invoke>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_classifies_noncanonical_wrapper_as_malformed(self):
        text = """I will tap the target.
<function_calls >
<invoke name="click"><parameter name="index">12</parameter></invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_classifies_missing_close_tag_as_malformed(self):
        text = """I will tap the target.
<function_calls>
<invoke name="click"><parameter name="index">12</parameter></invoke>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_classifies_empty_wrapper_as_malformed(self):
        result = parse_tool_calls_detailed("<function_calls>\n</function_calls>")

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_classifies_wrapper_without_named_invoke_as_malformed(self):
        text = """<function_calls>
<invoke><parameter name="index">12</parameter></invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_classifies_valid_xml_as_valid(self):
        text = """I will tap the target.
<function_calls>
<invoke name="click"><parameter name="index">12</parameter></invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.VALID)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, {"index": "12"})

    def test_malformed_first_block_is_not_replaced_by_valid_sibling(self):
        malformed = """<function_calls>
<｜｜DSML｜｜ name="click">
<parameter name="index">12</parameter>
</｜｜DSML｜｜>
</function_calls>"""
        text = f"""I will retry and then complete.
{malformed}
<function_calls>
<invoke name="complete">
<parameter name="success">true</parameter>
<parameter name="message">Done</parameter>
</invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.thought, "I will retry and then complete.")

    def test_returns_only_first_block_of_multi_block_reply(self):
        first = """<function_calls>
<invoke name="click"><parameter name="index">1</parameter></invoke>
</function_calls>"""
        text = f"""I will tap Open app.
{first}
The app has loaded.
<function_calls>
<invoke name="click"><parameter name="index">2</parameter></invoke>
</function_calls>
Done.
<function_calls>
<invoke name="complete"><parameter name="success">true</parameter></invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.VALID)
        self.assertEqual(result.thought, "I will tap Open app.")
        self.assertEqual(
            [(call.name, call.parameters) for call in result.calls],
            [("click", {"index": "1"})],
        )

    def test_single_block_returns_all_invocations(self):
        text = """I will tap and scroll.
<function_calls>
<invoke name="click"><parameter name="index">1</parameter></invoke>
<invoke name="swipe"><parameter name="index">2</parameter></invoke>
</function_calls>
"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual([call.name for call in result.calls], ["click", "swipe"])

    def test_xml_error_in_first_block_is_malformed(self):
        malformed = """<function_calls>
<invoke name="open_app"><parameter name="text">Settings</parameter>
</function_calls>"""
        text = f"""I will open Settings.
{malformed}
Settings is open.
<function_calls>
<invoke name="click"><parameter name="index">5</parameter></invoke>
</function_calls>
Done.
<function_calls>
<invoke name="complete"><parameter name="success">true</parameter></invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.thought, "I will open Settings.")

    def test_unclosed_first_block_is_malformed_before_valid_block(self):
        unclosed = """I will tap the target.
<function_calls>
<invoke name="click"><parameter name="index">12</parameter></invoke>
Tapped.
"""
        text = f"""{unclosed}<function_calls>
<invoke name="complete"><parameter name="success">true</parameter></invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_unclosed_only_block_is_malformed(self):
        text = """I will tap the target.
<function_calls>
<invoke name="click"><parameter name="index">12</parameter></invoke>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_trailing_prose_with_open_tags_does_not_change_first_block(self):
        first = """<function_calls>
<invoke name="click"><parameter name="index">1</parameter></invoke>
</function_calls>"""
        text = f"""I will tap Open app.
{first}
I used one <function_calls> block and will wait for its <function_calls> result."""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.VALID)
        self.assertEqual([call.name for call in result.calls], ["click"])

    def test_tag_mentioned_before_first_block_starts_the_first_block(self):
        block = """<function_calls>
<invoke name="click"><parameter name="index">1</parameter></invoke>
</function_calls>"""
        for prose in (
            "I will use a <function_calls> block now:",
            "Wrap calls in `<function_calls>` tags.",
        ):
            with self.subTest(prose=prose):
                text = f"{prose}\n{block}"

                result = parse_tool_calls_detailed(text)

                self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
                self.assertEqual(result.calls, [])

    def test_prose_mention_before_malformed_first_block_is_malformed(self):
        text = """I will use one <function_calls> block:
<function_calls>
<invoke name="click"><parameter name="index">1</parameter>
</function_calls>
<function_calls>
<invoke name="complete"><parameter name="success">true</parameter></invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.calls, [])

    def test_broken_first_block_wins_over_later_valid_block(self):
        later = """<function_calls>
<invoke name="click"><parameter name="index">2</parameter></invoke>
</function_calls>"""
        first_blocks = {
            "empty": "<function_calls>\n</function_calls>",
            "json": '<function_calls>\n{"name": "open_app"}\n</function_calls>',
            "unknown_tag": '<function_calls>\n<invoke_tool name="open_app"/>\n</function_calls>',
            "inline_json": 'Here: <function_calls>{"name": "open_app"}</function_calls>',
            "inline_empty": "Done: <function_calls></function_calls>",
            "inline_array": 'Here: <function_calls>[{"name": "click"}]</function_calls>',
            "inline_paren": "Here: <function_calls>(open_app Settings)</function_calls>",
            "inline_quoted": 'Here: <function_calls>"open_app"</function_calls>',
            "lone_surrogate": "<function_calls>\n\ud800\n</function_calls>",
            "inline_closed_words": "Here: <function_calls>click(index=1)</function_calls>",
            "closed_mention": "Calls go in <function_calls>...</function_calls> tags.",
            "inline_unclosed_words": "Here: <function_calls>click(index=1)",
            "stray_close_tag": "Done.</function_calls>",
            "full_width_stray_close_tag": "Done.＜／function_calls＞",
            "tool_calls_wrapper": '<tool_calls>{"name": "click"}</tool_calls>',
            "tool_call_wrapper": '<tool_call>{"name": "click"}</tool_call>',
            "function_call_wrapper": '<function_call name="click">1</function_call>',
        }
        for name, first in first_blocks.items():
            with self.subTest(name=name):
                result = parse_tool_calls_detailed(f"{first}\n{later}")

                self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
                self.assertEqual(result.calls, [])

    def test_alternate_wrapper_without_canonical_block_is_malformed(self):
        result = parse_tool_calls_detailed(
            'I will tap it.\n<tool_call>{"name": "click"}</tool_call>'
        )

        self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
        self.assertEqual(result.thought, "I will tap it.")

    def test_broken_call_inside_first_block_makes_it_malformed(self):
        complete = '<invoke name="complete">' '<parameter name="success">true</parameter></invoke>'
        first_calls = {
            "full_width_invoke": '＜invoke name="click"＞＜/invoke＞',
            "tool_call": '<tool_call>{"name": "click"}</tool_call>',
            "function_call": '<function_call name="click">1</function_call>',
            "nameless_invoke": '<invoke><parameter name="index">1</parameter></invoke>',
            "unknown_element": '<invoke_tool name="click"/>',
            "unknown_invoke_child": '<invoke name="click"><argument name="index">1</argument></invoke>',
            "nameless_parameter": '<invoke name="click"><parameter>1</parameter></invoke>',
            "text_call": "click(index=1)",
            "text_inside_invoke": '<invoke name="click">tap<parameter name="index">1</parameter></invoke>',
            "text_after_parameter": '<invoke name="click"><parameter name="index">1</parameter>x</invoke>',
            "comment": "<!-- click(index=1) -->",
            "processing_instruction": "<?call click(index=1)?>",
            "cdata": "<![CDATA[click(index=1)]]>",
            "element_in_parameter": (
                '<invoke name="type"><parameter name="text" kind="x">'
                "<argument>bad</argument></parameter></invoke>"
            ),
        }
        for name, first in first_calls.items():
            with self.subTest(name=name):
                result = parse_tool_calls_detailed(
                    f"<function_calls>\n{first}\n{complete}\n</function_calls>"
                )

                self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
                self.assertEqual(result.calls, [])

    def test_tool_markup_inside_parameter_value_is_text(self):
        text = """<function_calls>
<invoke name="type"><parameter name="text">use ＜invoke＞ or <tool_call> here</parameter></invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.VALID)
        self.assertEqual(result.calls[0].parameters, {"text": "use ＜invoke＞ or <tool_call> here"})

    def test_comment_like_text_inside_parameter_value_is_text(self):
        text = """<function_calls>
<invoke name="type"><parameter name="text">a <!-- b --> <?x?></parameter></invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.VALID)
        self.assertEqual(result.calls[0].parameters, {"text": "a <!-- b --> <?x?>"})

    def test_single_quoted_parameter_value_is_text(self):
        text = """<function_calls>
<invoke name="type"><parameter name='text'>a <b>x</b></parameter></invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.VALID)
        self.assertEqual(result.calls[0].parameters, {"text": "a <b>x</b>"})

    def test_non_canonical_first_block_is_not_replaced_by_later_block(self):
        first_blocks = {
            "dsml": """<｜DSML｜tool_calls>
<｜DSML｜invoke name="click">
<｜DSML｜parameter name="index">12</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>""",
            "full_width": """＜function_calls＞
＜invoke name="click"＞＜parameter name="index"＞12＜/parameter＞＜/invoke＞
＜/function_calls＞""",
            "bare_invoke": """<invoke name="click">
<parameter name="index">12</parameter>
</invoke>""",
        }
        for name, first in first_blocks.items():
            with self.subTest(name):
                prefix = f"I will tap the Profile tab.\n{first}\nProfile is open.\n"
                text = f"""{prefix}<function_calls>
<invoke name="complete"><parameter name="success">true</parameter></invoke>
</function_calls>"""

                result = parse_tool_calls_detailed(text)

                self.assertEqual(result.status, ToolCallParseStatus.MALFORMED)
                self.assertEqual(result.calls, [])
                self.assertEqual(result.thought, "I will tap the Profile tab.")

    def test_non_canonical_block_after_first_block_is_ignored(self):
        first = """<function_calls>
<invoke name="click"><parameter name="index">1</parameter></invoke>
</function_calls>"""
        text = f"""I will tap.
{first}
<｜DSML｜tool_calls>
<｜DSML｜invoke name="click"></｜DSML｜invoke>
</｜DSML｜tool_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.VALID)
        self.assertEqual([call.parameters for call in result.calls], [{"index": "1"}])

    def test_argument_error_is_valid_markup(self):
        text = """<function_calls>
<invoke name="click"><parameter name="index">not-a-number</parameter></invoke>
</function_calls>"""

        result = parse_tool_calls_detailed(text)

        self.assertEqual(result.status, ToolCallParseStatus.VALID)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, {"index": "not-a-number"})

    def test_drops_adjacent_exact_duplicate_tool_calls(self):
        text = """
I will tap the target.
<function_calls>
<invoke name="click_at">
<parameter name="x">128</parameter>
<parameter name="y">1560</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="click_at">
<parameter name="x">128</parameter>
<parameter name="y">1560</parameter>
</invoke>
</function_calls>
"""

        result = parse_tool_calls_detailed(text)
        thought, calls = result.thought, result.calls

        self.assertIn("I will tap", thought)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "click_at")
        self.assertEqual(calls[0].parameters, {"x": "128", "y": "1560"})

    def test_keeps_non_identical_sequential_calls(self):
        text = """
I will tap two different targets.
<function_calls>
<invoke name="click_at">
<parameter name="x">128</parameter>
<parameter name="y">1560</parameter>
</invoke>
<invoke name="click_at">
<parameter name="x">200</parameter>
<parameter name="y">1560</parameter>
</invoke>
</function_calls>
"""

        calls = parse_tool_calls_detailed(text).calls

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].parameters, {"x": "128", "y": "1560"})
        self.assertEqual(calls[1].parameters, {"x": "200", "y": "1560"})

    def test_keeps_identical_invokes_inside_one_block(self):
        text = """
I will press back twice.
<function_calls>
<invoke name="system_button">
<parameter name="button">back</parameter>
</invoke>
<invoke name="system_button">
<parameter name="button">back</parameter>
</invoke>
</function_calls>
"""

        calls = parse_tool_calls_detailed(text).calls

        self.assertEqual(len(calls), 2)
        self.assertEqual([call.name for call in calls], ["system_button", "system_button"])
        self.assertEqual(calls[0].parameters, {"button": "back"})
        self.assertEqual(calls[1].parameters, {"button": "back"})

    def test_keeps_intentional_mixed_batch(self):
        text = """
I will focus the field and type.
<function_calls>
<invoke name="click_at">
<parameter name="x">261</parameter>
<parameter name="y">1888</parameter>
</invoke>
<invoke name="type_text">
<parameter name="text">Android version</parameter>
<parameter name="clear">true</parameter>
</invoke>
</function_calls>
"""

        calls = parse_tool_calls_detailed(text).calls

        self.assertEqual([call.name for call in calls], ["click_at", "type_text"])
        self.assertEqual(calls[0].parameters, {"x": "261", "y": "1888"})
        self.assertEqual(
            calls[1].parameters,
            {"text": "Android version", "clear": "true"},
        )

    def test_duplicate_complete_blocks_execute_once(self):
        text = """
The task is done.
<function_calls>
<invoke name="complete">
<parameter name="success">true</parameter>
<parameter name="message">Done</parameter>
</invoke>
</function_calls>
<function_calls>
<invoke name="complete">
<parameter name="success">true</parameter>
<parameter name="message">Done</parameter>
</invoke>
</function_calls>
"""

        calls = parse_tool_calls_detailed(text).calls

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "complete")
        self.assertEqual(calls[0].parameters, {"success": "true", "message": "Done"})

    def test_extract_add_memory_basic(self):
        text = "I see the email.\n<add_memory>Meeting at 3pm Thursday Room 204</add_memory>\nNow I'll click reply."
        result = extract_add_memory(text)
        self.assertEqual(result, "Meeting at 3pm Thursday Room 204")

    def test_extract_add_memory_empty(self):
        text = "Just a thought with no memory tag."
        result = extract_add_memory(text)
        self.assertEqual(result, "")

    def test_extract_add_memory_whitespace(self):
        text = "<add_memory>  spaced content  </add_memory>"
        result = extract_add_memory(text)
        self.assertEqual(result, "spaced content")

    def test_extract_add_memory_multiline(self):
        text = """Some thought here.
<add_memory>
Line 1: Meeting at 3pm
Line 2: Room 204
</add_memory>
Tool calls follow."""
        result = extract_add_memory(text)
        self.assertIn("Meeting at 3pm", result)
        self.assertIn("Room 204", result)

    def test_extract_add_memory_with_tool_calls(self):
        text = """I see the password field.
<add_memory>Username is admin@test.com</add_memory>
<function_calls>
<invoke name="click"><parameter name="index">5</parameter></invoke>
</function_calls>"""
        result = parse_tool_calls_detailed(text)
        thought, calls = result.thought, result.calls
        memory = extract_add_memory(thought)
        self.assertEqual(memory, "Username is admin@test.com")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "click")

    def test_extract_add_memory_multiple_blocks(self):
        text = """I found two important things on this screen.
<add_memory>User email is a@example.com</add_memory>
<add_memory>Verification code is 123456</add_memory>"""
        result = extract_add_memory(text)
        self.assertIn("User email is a@example.com", result)
        self.assertIn("Verification code is 123456", result)


if __name__ == "__main__":
    unittest.main()
