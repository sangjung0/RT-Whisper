from dataclasses import dataclass

from rt_whisper.data import Context, Sentence, Token


@dataclass(slots=True)
class ComposerResult:
    completed: list[Sentence]
    candidate: list[Sentence]
    recycle_completed_tokens: list[Token]
    order: int

    def update_context(self, context: Context):
        context.completed = self.completed
        context.candidate = self.candidate
        context.recycle_completed_tokens = self.recycle_completed_tokens
        context.order = self.order
