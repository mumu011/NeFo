from .policy import QuestionSample as BaseQuestionSample
from utils import is_none
import shortuuid

class CleanQuestionSample(BaseQuestionSample):
    def __init__(self, row, args, round_idx=0):
        super().__init__(row, args, round_idx)

    async def get_final_answer(self):
        final_qs = ''
        
        # Add hint information
        if not is_none(self.row['hint']):
            final_qs += self.row['hint'] + '\n'
            
        final_qs += self.row['question']
        
        # Add options
        for option_char, option in zip(self.cur_option_char, self.options):
            final_qs += '\n' + option_char + '. ' + option

        if self.args.single_pred_prompt:
            if self.args.lang == 'cn':
                final_qs += '\n' + "请直接回答选项字母。"
            else:
                final_qs += '\n' + "Answer with the option's letter from the given choices directly."

        answer = await self.generate(final_qs, self.image)
        
        # Filter answer to extract option letter
        for letter in ['A', 'B', 'C', 'D']:
            if letter in answer:
                return letter, final_qs
        return 'A', final_qs  # Default return A

    async def _process(self):
        final_answer, prompt = await self.get_final_answer()
        
        return {
            "question_id": self.row['index'],
            "round_id": self.round_idx,
            "prompt": prompt,
            "text": final_answer,
            "options": self.options,
            "option_char": self.cur_option_char,
            "answer": self.row['answer'],
            "answer_id": shortuuid.uuid(),
            "model_id": self.args.model_path,
            "metadata": {}
        }
