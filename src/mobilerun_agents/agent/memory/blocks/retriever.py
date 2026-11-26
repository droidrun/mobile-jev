from llama_index.core.retrievers import BaseRetriever

from mobilerun_agents.agent.memory.memory import SnapshotBlock
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot


class RetrieverBlock(SnapshotBlock):
    name: str = "retriever"
    priority: int = 1
    retriever: BaseRetriever

    async def render(self, snapshot: StepSnapshot) -> str:
        phone = snapshot.ui.phone_state
        query = "\n".join(
            filter(None, (snapshot.goal, phone.get("packageName"), phone.get("activityName")))
        )
        nodes = await self.retriever.aretrieve(query)
        return "\n\n".join(node.get_content() for node in nodes)
