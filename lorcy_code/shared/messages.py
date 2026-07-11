

# 消息分组
def _group_messages_by_turn(messages: list) -> list[list]:
    """
    将消息按轮次分组（参考 chagent 逻辑）
    从一个 HumanMessage 开始，到下一个 HumanMessage 之前为一组
    """
    groups = []
    current_group = []

    for msg in messages:
        if msg.type == "human":  # 下一组消息的第一个消息：HumanMessage
            if current_group:  # 当前消息组
                groups.append(current_group)
            current_group = [msg]  # 把下一组消息的第一个消息：HumanMessage，放入新的消息组
        else:
            current_group.append(msg)  # 把下一组消息的其余消息也放入新的消息组

    if current_group:  # 所有消息都遍历完 还没放入消息组
        groups.append(current_group)  # 所以需要放入消息组

    return groups

# 历史会话的会话名显示
def _get_group_display(group: list) -> str:
    """获取消息组的显示文本（以 HumanMessage 内容为代表）"""
    from lorcy_code.shared.text import get_text_content
    for msg in group: # 遍历消息组
        if msg.type == "human": # 遇到HumanMessage的话
            text_content = get_text_content(msg.content)   # 获取消息文本内容前60字当场会话名显示
            content = text_content[:60].replace("\n", " ")
            if len(text_content) > 60:
                content += "..."
            return content
    return "(空消息组)"

# 收集即将被压缩的消息的消息id组
def _collect_ids_from_group(group_index: int, groups: list) -> tuple[list[str], list[str]]:
    all_ids = [m.id for group in groups for m in group]
    no_need_ids = []
    for i, group in enumerate(groups):
        if i >= group_index:
            no_need_ids.extend([m.id for m in group])
    return no_need_ids, all_ids
