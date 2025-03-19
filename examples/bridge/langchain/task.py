from agent import web_research_agent

from _8AI import Task, task
from _8AI.dataset import json_dataset
from _8AI.scorer import model_graded_fact
from _8AI.solver import bridge


@task
def research() -> Task:
    return Task(
        dataset=json_dataset("dataset.json"),
        solver=bridge(web_research_agent()),
        scorer=model_graded_fact(),
    )
