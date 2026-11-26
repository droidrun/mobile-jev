from llama_index.core.retrievers import BaseRetriever

from mobilerun_agents.agent.memory.memory import SnapshotBlock
from mobilerun_agents.agent.memory.snapshot import StepSnapshot


class RetrieverBlock(SnapshotBlock):
    name: str = "retriever"
    priority: int = 1
    retriever: BaseRetriever

    async def render(self, snapshot: StepSnapshot) -> str:
        query = "\n".join(filter(None, (snapshot.goal, snapshot.package, snapshot.activity)))
        nodes = await self.retriever.aretrieve(query)
        return "\n\n".join(node.get_content() for node in nodes)
