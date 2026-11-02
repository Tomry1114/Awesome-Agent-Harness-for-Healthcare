Complete the requested medical workflow using the current browser screenshot. Use BrowserAction with exactly one action from the supplied reference per call. Coordinate actions and keyboard input are available; DOM, selectors and element identifiers are not. Observe the new screenshot before deciding the next action. Submit any answer needed by the environment using send_msg_to_user(text). A final prose answer alone does not submit it. Stop when the environment terminates or truncates. Do not infer medical correctness merely from a successful click.

Tools:
{tools}
Use <tool_call>{"name":"BrowserAction","arguments":{"action":"ACTION"}}</tool_call> for a tool call and <answer>TEXT</answer> to finish.
