import numpy as np
from vllm import LLM
import jsonlines
from transformers import AutoTokenizer
import os
from pathlib import Path
from tqdm import tqdm
import sys
import argparse
import re
import time

from contriever_model import load_contriever_and_tokenizer
from my_retriever import DenseRetriever
from paths import default_run_name
from utils import chat_vllm
from vllm_lora import any_lora_paths, llm_lora_init_kwargs, make_lora_request

STRIDE_ROOT = Path(__file__).resolve().parent  # code / repository root (flat layout)


### 从meta_plan中提取Concrete Plan
def extract_plans(meta_plan):
    try:
        need_str = re.findall(r'(Concrete Plan:.*)', meta_plan, re.DOTALL)[0].strip()
        need_str = need_str.replace('Concrete Plan:', 'Plan:')
        return need_str
    except IndexError:
        return meta_plan.strip()
    
def check_none_answer(answer):
    answer = answer.strip().lower() if isinstance(answer, str) else answer
    if answer is None:
        return True
    if isinstance(answer, str) and answer in ['none', 'no ', 'n/a', 'not mentioned', 'not given', 'unknown', '']:
        return True
    if isinstance(answer, list) and all(isinstance(a, str) and a.strip().lower() in ['none', 'no', 'n/a', 'not mentioned', 'not given', 'unknown', ''] for a in answer):
        return True
    if isinstance(answer, str) and re.search(r'\b(none|no |n/a|not mentioned|not given|unknown)\b', answer):
        return True
    return False

def chunk_list(lst, chunk_size):
    """将列表 lst 按 chunk_size 分块"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default='0', type=str, help="gpu id"
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Generative model: Hugging Face model id or local directory for vLLM",
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        type=str,
        help="Path to input jsonl (same schema as meta-plan stage; id, question, ...)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Subfolder under meta_plans/ and output/ (default: stem of input_jsonl); must match meta-plan run",
    )
    parser.add_argument(
        "--index_corpus",
        type=str,
        default=None,
        help="Replaces the substring 'dataset' in --faiss_index_path (default: same as run_name)",
    )
    parser.add_argument(
        '--plan_file_path',
        type=str,
        default=None,
        help="meta plan root (default: meta_plans)",
    )
    parser.add_argument(
        '--plan_file_name', type=str, default='meta_plan.jsonl', help="meta plan filename under meta_plans/<run_name>/",
    )
    parser.add_argument(
        "--write_file_name", default='stride', type=str, help="output base name"
    )
    parser.add_argument(
        '--s_prompt_file', type=str, default='default', help="prompt/supervisor/<stem>.txt"
    )
    parser.add_argument(
        '--e_prompt_file', type=str, default='default', help="extractor prompt stem"
    )
    parser.add_argument(
        '--r_prompt_file', type=str, default='default', help="reasoner prompt stem"
    )
    parser.add_argument(
        '--top_k_docs', type=int, default=5, help="top k retrieved documents"
    )
    parser.add_argument(
        '--retriever_model_path',
        type=str,
        default='facebook/contriever',
        help="Hugging Face model id or local path (Contriever family)",
    )
    parser.add_argument(
        '--faiss_index_path',
        type=str,
        default=None,
        help="path with literal 'dataset' (replaced by corpus name); default: faiss_index/dataset/index",
    )
    parser.add_argument(
        '--max_iteration', type=int, default=5, help="max iteration for supervisor"
    )
    parser.add_argument(
        '--run_data_num', type=int, default=-1, help="Max examples (-1 = all)"
    )
    parser.add_argument(
        '--use_qwen3', type=bool, default=False, help="Whether to use qwen3"
    )
    parser.add_argument(
        '--think_mode', type=bool, default=False, help="Whether to use think mode for qwen3"
    )
    parser.add_argument(
        '--failed_threshold', type=int, default=2, help="Number of times a sub-question can fail before giving up"
    )
    parser.add_argument(
        '--bs_per_iter', type=int, default=4, help="batch size per iteration for batch processing"
    )
    ### VLLM 参数
    parser.add_argument(
        '--max_model_len', default=8192, type=int, help='max_model_len (prompt+output) for Vllm init'
    )
    parser.add_argument(
        '--max_num_seqs', default=64, type=int, help='max_num_seqs for Vllm init'
    )
    parser.add_argument(
        '--gpu_memory_utilization', default=0.85, type=float, help='gpu_memory_utilization for Vllm init'
    )
    parser.add_argument(
        '--tensor_parallel_size', default=1, type=int, help='tensor_parallel_size for Vllm init'
    )
    parser.add_argument(
        "--lora_supervisor",
        type=str,
        default=None,
        help="PEFT adapter for Supervisor module",
    )
    parser.add_argument(
        "--lora_extractor",
        type=str,
        default=None,
        help="PEFT adapter for Extractor module",
    )
    parser.add_argument(
        "--lora_reasoner",
        type=str,
        default=None,
        help="PEFT adapter for Reasoner module",
    )
    parser.add_argument("--max_lora_rank", type=int, default=64, help="vLLM max LoRA rank")
    parser.add_argument("--max_loras", type=int, default=8, help="vLLM max concurrent LoRA adapters")
    args = parser.parse_args()
    run_name = args.run_name if args.run_name else default_run_name(args.input_jsonl)
    index_corpus = args.index_corpus if args.index_corpus else run_name
    if args.plan_file_path is None:
        args.plan_file_path = str(STRIDE_ROOT / "meta_plans")
    if args.faiss_index_path is None:
        args.faiss_index_path = str(STRIDE_ROOT / "faiss_index" / "dataset" / "index")

    '''
    Supervisor: iteratively chooses actions (retrieve / answer / rewrite), calls
    Extractor + Reasoner, tracks solved vs pending sub-questions until done or failure threshold.
    Reasoner output is parsed as JSON (``answer`` field); failures use ``check_none_answer``.
    '''
    
    batch_size = args.batch_size
    model_path = args.model_path
    use_lora = any_lora_paths(
        args.lora_supervisor,
        args.lora_extractor,
        args.lora_reasoner,
    )

    model = LLM(
        model=model_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        **llm_lora_init_kwargs(
            use_lora=use_lora,
            max_lora_rank=args.max_lora_rank,
            max_loras=args.max_loras,
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    req_s = make_lora_request("supervisor", 2, args.lora_supervisor)
    req_e = make_lora_request("extractor", 3, args.lora_extractor)
    req_r = make_lora_request("reasoner", 4, args.lora_reasoner)
    
        
    ### supervisor prompt文件
    s_prompt_path = STRIDE_ROOT / "prompt" / "supervisor" / f"{args.s_prompt_file}.txt"
    with open(s_prompt_path, "r", encoding="utf-8") as f:
        s_full_prompt = f.read()

    ### extractor prompt文件
    e_prompt_path = STRIDE_ROOT / "prompt" / "extractor" / f"{args.e_prompt_file}.txt"
    with open(e_prompt_path, "r", encoding="utf-8") as f:
        e_full_prompt = f.read()

    ### reasoner prompt文件
    r_prompt_path = STRIDE_ROOT / "prompt" / "reasoner" / f"{args.r_prompt_file}.txt"
    with open(r_prompt_path, "r", encoding="utf-8") as f:
        r_full_prompt = f.read()

    ### qwen3
    if 'Qwen3' in model_path:
        args.use_qwen3 = True
    else:
        args.think_mode = None
        
    ### 写入文件的文件名
    write_file_name = args.write_file_name

    meta_plan_version = args.plan_file_name.split('_')[-1].replace('.jsonl', '')

    write_file_name = f'{write_file_name}_top{args.top_k_docs}'
    if args.max_iteration != 5:
        write_file_name = write_file_name + f'-iter{args.max_iteration}'
    if args.failed_threshold != 2:
        write_file_name = write_file_name + f'-f{args.failed_threshold}'

    
    ### 加载检索模型和faiss index
    used_contriever, used_contriever_tokenizer = load_contriever_and_tokenizer(
        args.retriever_model_path
    )
    
    used_retriever = DenseRetriever(used_contriever, used_contriever_tokenizer)
    
    faiss_index_path = args.faiss_index_path.replace('dataset', index_corpus)
    used_retriever.load_index(faiss_index_path)
    
    note = ''
    note += f', 使用{args.model_path}模型, batch_size={batch_size}'
    note += f', top_k_docs={args.top_k_docs}, faiss_index_path={faiss_index_path}, 文档数={used_retriever.ctr}'
    note += f', plan_file={args.plan_file_path}/{run_name}/{args.plan_file_name}'
    note += f', supervisor prompt={args.s_prompt_file}, extractor prompt={args.e_prompt_file}, reasoner prompt={args.r_prompt_file}'
    
    
    print(note)
    
    path = str(Path(args.input_jsonl).expanduser().resolve())
    with jsonlines.open(path) as reader:
        total_data = [item for item in reader]
    total_data = total_data[:args.run_data_num] if args.run_data_num > 0 else total_data
            
    ### 读取meta plan
    plan_path = os.path.join(args.plan_file_path, run_name, args.plan_file_name)
    plan_dict = {}
    with jsonlines.open(plan_path, 'r') as reader:
        for item in reader:
            plan = extract_plans(item['predict'])
            plan_dict[item['id']] = plan

    ### 写入文件的path
    write_path = str(
        STRIDE_ROOT / "output" / run_name / meta_plan_version / f"{write_file_name}.jsonl"
    )
    if not os.path.exists(os.path.dirname(write_path)):
        os.makedirs(os.path.dirname(write_path))
    
    exist_data = []
    if os.path.exists(write_path):
        with jsonlines.open(write_path) as reader:
            for obj in reader:
                exist_data.append(obj['id'])
    print(f"Already exist {len(exist_data)} sentences in {write_path}!")

    output_folder = str(STRIDE_ROOT / "output" / run_name / meta_plan_version / "log")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 新建txt文件并指定路径
    output_file = os.path.join(output_folder, f'{write_file_name}.txt')
    
    # 打开文件准备写入内容
    print("Ready to process main task!")
    with open(output_file, 'a') as f:
        # 重定向标准输出到txt文件
        sys.stdout = f
        print(note)
        with jsonlines.open(write_path, 'a') as writer:
            for item in tqdm(total_data, desc=f"Supervisor + E+R [{run_name}]"):
                id_ = item['id']
                if id_ in exist_data:
                    continue
                
                question = item['question']
                current_plan = plan_dict[id_]

                total_tokens = {"supervisor": [[], []], "extractor": [[], []], 'reasoner': [[], []]} # input, output
                total_time = {"supervisor": [], "extractor": [], 'reasoner': []}
                
                print(f"\n================== Start Processing ID: {id_} ==================\n")
                print(f"Question: \n{question}\n")

                # messages = []
                
                ### 1. 构造Supervisor的输入
                current_plan.replace('Concrete Plan', 'Plan')
                max_qid = int(re.findall(r'Q\d+: ', current_plan)[-1].strip().replace("Q", "").replace(":","")) ### 当前plan中最大的qid
                solved_questions = {} ### 记录已经解决的 qid: answer
                pending_qids = [f"Q{i}" for i in range(1, max_qid + 1)] ### 记录当前还没有解决的qid
                
                for qid in pending_qids: ### 如果plan中没有出现的qid，去掉
                    if qid not in current_plan:
                        pending_qids.remove(qid)
                
                failure_log = {}
                progress_str = "Solved: {}\nPending: [" + ', '.join(pending_qids) + "]\nFailureLog: {}"

                s_post = f"Question: {question}\n\n{current_plan}\n\nProgress: \n{progress_str}"

                messages = [
                    {"role": "system", "content": s_full_prompt},
                    {"role": "user", "content": s_post},
                ]
                iter_num = 3
                s_output = None
                s_output_list = []
                while iter_num > 0:
                    s_output, input_length, output_length, time = chat_vllm(
                        messages,
                        model,
                        tokenizer,
                        qwen3_think_mode=args.think_mode,
                        lora_request=req_s,
                    )
                    ### 如果是list且不是batch输入的，则取[0] (chat_vllm返回的是list)
                    if isinstance(s_output, list):
                        s_output, input_length, output_length = s_output[0], input_length[0], output_length[0]
                    try:
                        s_output = re.findall(r'```json(.*?)```', s_output, re.DOTALL)[0].strip()
                        # s_output_list = s_output.replace("\n", " ")
                        s_output_list = re.sub(r'// .*', '', s_output) ### 去掉注释
                        s_output_list = eval(s_output_list) ### 变成list,元素是dict
                        if isinstance(s_output_list, str):
                            print(f"Failed to parse Supervisor output json at Iteration {iteration}.")
                            print(f"Raw Supervisor output:\n{s_output}\n")
                            iter_num -= 1
                            continue
                    except Exception as e:
                        print(f"Error in Supervisor output parsing, try again! Error: {e}")
                        print(f"Raw Supervisor output:\n{s_output}\n")
                        iter_num -= 1
                        continue
                    if isinstance(s_output_list, str):
                        print(f"Failed to convert Supervisor output to json, try again! Raw output:\n{s_output}\n")
                        iter_num -= 1
                        continue
                    break
                if s_output is None: ### 多次尝试后，仍然没有得到有效的输出，暂时跳过，后续处理
                    print(f"Failed to get valid output from Supervisor for ID: {id_}, skip this case.")
                    continue
                
                total_tokens['supervisor'][0].append(input_length)
                total_tokens['supervisor'][1].append(output_length)
                total_time['supervisor'].append(time)
                
                print(f"\n================== Iteration 1 ==================\n")
                print("\nSupervisor round 1:")
                print(f"Input: \n{s_post}\n\nOutput: \n{s_output}\n")
                print('-------------------------------------------')
                print(f'Input tokens: {input_length}, Output tokens: {output_length}')
                print(f'Time: {time} seconds\n')
                print('-------------------------------------------')
                
                if isinstance(s_output_list, str):
                    print(f"Failed to parse Supervisor output json at Iteration 1, skip this case.")
                    continue
                
                ### 进入迭代
                final_answer = None
                iteration = 1
                extract_facts = [] ### 记录每一步Extractor得到的fact，（query， fact）
                supervisor_records = {"progress": [progress_str], "output": [s_output]} ### 记录每一步Supervisor的输入，和输出
                total_retrieval_results = []
                reasoner_records = [] ### 记录每一步Reasoner的输入，和输出
                fact_dict = {} ### key是 qid, value是 (query, fact), 记录本次过程中所有得到的fact
                FINAL_ANSWER_FLAG = False ### 标记是否已经得到最终答案
                FAILED_FLAG = False ### 标记当前case是否失败，无法继续
                while final_answer is None and iteration <= args.max_iteration:
                    iteration += 1
                    # 假设当前正在处理一个主问题 id_
                    # s_output_list 来自 Supervisor 的输出（当前 iteration）

                    # ===== 阶段1：收集所有 retrieve/rewrite 类子问题，用于批量检索 + Extractor =====
                    extractor_tasks = []  # [(qid, action, retrieval_query), ...]
                    
                    MAX_BATCH_SIZE = args.bs_per_iter  # 比如 4 或 8，通过命令行参数控制

                    for s_ in s_output_list:
                        if isinstance(s_, str):
                            continue
                        try:
                            qid = s_['qid']
                            action = s_['action']
                            retrieval_query = s_['query']
                        except Exception as e:
                            print(f"Error in Supervisor output parsing for Iteration {iteration}, skip this sub-output:\n{s_}\n")
                            continue

                        # 检查是否已失败太多次
                        if qid in failure_log and len(failure_log[qid]) >= args.failed_threshold:
                            print(f"Sub-question {qid} already failed {args.failed_threshold} times, skip.")
                            FAILED_FLAG = True
                            break

                        if action in ['retrieve', 'rewrite']:
                            extractor_tasks.append((qid, action, retrieval_query))
                        elif action == 'answer':
                            # answer 类不能在此阶段处理，留到 Reasoner 阶段（需等所有 facts）
                            pass
                        else:
                            print(f"Unknown action: {action} for qid {qid}, skip.")
                            continue
                    
                    # === 批量调用 Extractor ===
                        
                    if extractor_tasks and not FAILED_FLAG:
                        for chunk in chunk_list(extractor_tasks, MAX_BATCH_SIZE):
                            # === 对当前 chunk 批量检索 ===
                            queries = [task[2] for task in chunk]
                            all_retrieval_results = used_retriever.batch_retrieve(queries, top_k=args.top_k_docs)

                            # === 构造 prompts ===
                            extractor_prompts = []
                            doc_strs = []
                            chunk_retrieval_pairs = []  # 临时记录，用于后续更新 fact_dict
                            for i, (qid, action, retrieval_query) in enumerate(chunk):
                                retrieval_result = all_retrieval_results[i]
                                doc_str = ""
                                retrieval_results = []
                                for res in retrieval_result:
                                    title = res['title']
                                    text = res['text']
                                    doc_str += f"Title: {title}\n{text}\n\n"
                                    unique_id = f"{title}~~~{text[:20]}"
                                    retrieval_results.append(unique_id)
                                doc_strs.append(doc_str)
                                total_retrieval_results.append((retrieval_query, retrieval_results))
                                chunk_retrieval_pairs.append((qid, retrieval_query, doc_str))

                                e_post = f"Question: \n{retrieval_query}\n\nDocuments: \n{doc_str}"
                                messages = [
                                    {"role": "system", "content": e_full_prompt},
                                    {"role": "user", "content": e_post},
                                ]
                                extractor_prompts.append(messages)
                            
                            e_outputs, input_lengths, output_lengths, times = chat_vllm(
                                extractor_prompts,
                                model,
                                tokenizer,
                                qwen3_think_mode=args.think_mode,
                                lora_request=req_e,
                            )
                            time_used = times/len(e_outputs)

                            # === 处理当前 chunk 的输出 ===
                            for i, (qid, retrieval_query, doc_str) in enumerate(chunk_retrieval_pairs):
                                e_output = e_outputs[i].strip().replace("\n", "").replace("][", ", ")
                                input_len = input_lengths[i]
                                output_len = output_lengths[i]

                                total_tokens['extractor'][0].append(input_len)
                                total_tokens['extractor'][1].append(output_len)
                                total_time['extractor'].append(time_used)

                                print(f"[Extractor] qid={qid}")
                                print(f"Input: \nQuestion: \n{retrieval_query}\n\nDocuments: \n{doc_str}")
                                print(f"Output: \n{e_output}\n")
                                print('-------------------------------------------')
                                print(f'Input tokens: {input_len}, Output tokens: {output_len}')
                                print(f'Time: {time_used} seconds\n')
                                print('-------------------------------------------\n')

                                extract_facts.append((retrieval_query, e_output))
                                # 判断是否失败
                                if e_output is None or e_output == 'None':
                                    if qid not in failure_log:
                                        failure_log[qid] = []
                                    failure_log[qid].append(retrieval_query)
                                    if len(failure_log[qid]) >= args.failed_threshold:
                                        FAILED_FLAG = True
                                    continue
                                # 成功：更新 fact_dict
                                fact_dict[qid] = (retrieval_query, e_output)

                            if FAILED_FLAG:
                                break  # 跳出 chunk 循环

                    # ===== 阶段2：收集所有需要 Reasoner 的子问题 =====
                    reasoner_tasks = []  # [(qid, action, retrieval_query, facts_str), ...]

                    # 1. 先处理 retrieve/rewrite 成功的子问题（需要 sub-answer）
                    for (qid, action, retrieval_query) in extractor_tasks:
                        if qid in fact_dict:  # 说明 extractor 成功
                            facts_str = fact_dict[qid][1]  # e_output
                            reasoner_tasks.append((qid, action, retrieval_query, facts_str))

                    # 2. 再处理 action == 'answer' 的子问题（需要所有 facts）
                    for s_ in s_output_list:
                        if isinstance(s_, str):
                            continue
                        try:
                            qid = s_['qid']
                            action = s_['action']
                            retrieval_query = s_['query']
                        except:
                            continue
                        if action == 'answer':
                            # 构造所有 facts 字符串
                            fact_str = "\n".join(v[1] for v in fact_dict.values()).strip()
                            reasoner_tasks.append((qid, action, retrieval_query, fact_str))
                    
                    if reasoner_tasks and not FAILED_FLAG:
                        for chunk in chunk_list(reasoner_tasks, MAX_BATCH_SIZE):
                            reasoner_prompts = []
                            chunk_info = []  # [(qid, action, retrieval_query, facts_str), ...]
                            for ch in chunk:
                                qid, action, retrieval_query, facts_str = ch
                                r_post = f"Facts: \n{facts_str}\n\nQuestion: \n{retrieval_query}"
                                messages = [
                                    {"role": "system", "content": r_full_prompt},
                                    {"role": "user", "content": r_post},
                                ]
                                reasoner_prompts.append(messages)
                                chunk_info.append(ch)

                            r_outputs, r_input_lens, r_output_lens, r_times = chat_vllm(
                                reasoner_prompts,
                                model,
                                tokenizer,
                                qwen3_think_mode=args.think_mode,
                                lora_request=req_r,
                            )
                            r_time = r_times/len(r_outputs)

                            # 处理输出
                            for i, (qid, action, retrieval_query, facts_str) in enumerate(chunk_info):
                                r_output = r_outputs[i]
                                r_input_length = r_input_lens[i]
                                r_output_length = r_output_lens[i]

                                r_answer = None
                                r_failed_flag = False

                                try:
                                    if len(re.findall(r"```", r_output, re.DOTALL)) < 2:
                                        r_answer = re.findall(r"({.*})", r_output, re.DOTALL)[0].strip()
                                    else:
                                        r_answer = re.findall(
                                            r"```json(.*?)```", r_output, re.DOTALL
                                        )[0].strip()
                                    r_answer = re.sub(r"// .*", "", r_answer)
                                    r_answer = eval(r_answer)["answer"]
                                except Exception as e:
                                    print(f"Failed to extract answer from Reasoner output for qid {qid}, output: {r_output}")
                                    r_failed_flag = True
                                # 判断是否回答
                                if not r_failed_flag and isinstance(r_answer, str):
                                    if check_none_answer(r_answer):
                                        r_answer = None
                                        r_failed_flag = True
                                        print(f"Reasoner indicates not enough information for qid {qid}\noutput: {r_output}")
                                # 回答失败
                                if r_failed_flag:
                                    if qid not in failure_log:
                                        failure_log[qid] = []
                                    failure_log[qid].append(retrieval_query)
                                    if len(failure_log[qid]) >= args.failed_threshold:
                                        print(f"Sub-question {qid} failed more than {args.failed_threshold} times.")
                                        FAILED_FLAG = True
                                    continue

                                # 成功：记录
                                r_input = facts_str + "\n\nQuestion: \n" + retrieval_query
                                reasoner_records.append((qid, r_input, r_output, r_answer))
                                solved_questions[qid] = r_answer

                                if qid in pending_qids:
                                    pending_qids.remove(qid)
                                if qid in failure_log:
                                    failure_log.pop(qid)
                                
                                print(f"[Reasoner] qid={qid}")
                                print(f"Facts: \n{r_input}\n\nOutput: \n{r_output}\n\nParsed Answer: {r_answer}")
                                print('-------------------------------------------')
                                print(f'Input tokens: {r_input_length}, Output tokens: {r_output_length}')
                                print(f'Time: {r_time} seconds\n')
                                print('-------------------------------------------\n')
                                

                                print(f"Sub-question {qid} solved with answer: {r_answer}")
                                print(f"Current solved: {solved_questions}\nPending: {pending_qids}\n")

                                total_tokens['reasoner'][0].append(r_input_length)
                                total_tokens['reasoner'][1].append(r_output_length)
                                total_time['reasoner'].append(r_time)

                                # 检查是否最终答案
                                if qid == f"Q{max_qid}" or len(pending_qids) == 0:
                                    final_answer = r_answer
                                    FINAL_ANSWER_FLAG = True
                                    break  # 可跳出当前 reasoner 循环，但 batch 已执行完，影响不大

                            if FAILED_FLAG or FINAL_ANSWER_FLAG:
                                break
                    
                    # 如果在 reasoner 中触发了最终答案，可考虑 break 外层循环
                    if FINAL_ANSWER_FLAG:    
                        break
                    if FAILED_FLAG:
                        break

                    ### 构造下一轮Supervisor的输入
                    progress_str = "Solved: " + str(solved_questions) + "\nPending: [" + ', '.join(pending_qids) + "]\nFailureLog: " + str(failure_log)

                    supervisor_records['progress'].append(progress_str)
                    
                    s_post = f"Question: {question}\n\n{current_plan}\n\nProgress: \n{progress_str}"

                    messages = [
                        {"role": "system", "content": s_full_prompt},
                        {"role": "user", "content": s_post},
                    ]

                    iter_num = 3
                    s_output = None
                    s_output_list = {}
                    print(f"\n================== Iteration {iteration} ==================\n")
                    while iter_num > 0:
                        s_output, input_length, output_length, time = chat_vllm(
                            messages,
                            model,
                            tokenizer,
                            qwen3_think_mode=args.think_mode,
                            lora_request=req_s,
                        )
                        ### 如果是list且不是batch输入的，则取[0] (chat_vllm返回的是list)
                        if isinstance(s_output, list):
                            s_output, input_length, output_length = s_output[0], input_length[0], output_length[0]
                        try:
                            s_output = re.findall(r'```json(.*?)```', s_output, re.DOTALL)[0].strip()
                            s_output_list = re.sub(r'// .*', '', s_output) ### 去掉注释
                            s_output_list = eval(s_output_list) ### 变成list,元素是dict
                            if isinstance(s_output_list, str):
                                print(f"Failed to parse Supervisor output json at Iteration {iteration}.")
                                print(f"Raw Supervisor output:\n{s_output}\n")
                                iter_num -= 1
                                continue
                        except Exception as e:
                            print(f"Error in Supervisor output parsing, try again! Error: {e}")
                            print(f"Raw Supervisor output:\n{s_output}\n")
                            iter_num -= 1
                            continue
                        if isinstance(s_output_list, str):
                            print(f"Failed to convert Supervisor output to json, try again! Raw output:\n{s_output}\n")
                            iter_num -= 1
                            continue
                        break
                    if s_output is None: ### 多次尝试后，仍然没有得到有效的输出，暂时跳过，后续处理
                        print(f"Failed to get valid output from Supervisor for ID: {id_}, skip this case.")
                        continue
                    
                    total_tokens['supervisor'][0].append(input_length)
                    total_tokens['supervisor'][1].append(output_length)
                    total_time['supervisor'].append(time)
                    
                    print(f"\nSupervisor round {iteration}:\n")
                    print(f"Input: \n{s_post}\n\nOutput: \n{s_output}\n")
                    print('-------------------------------------------')
                    print(f'Input tokens: {input_length}, Output tokens: {output_length}')
                    print(f'Time: {time} seconds\n')
                    print('-------------------------------------------')
                    
                    supervisor_records['output'].append(s_output)
                    
     
                if final_answer is None:
                    print(f"Failed to get final answer within max iterations for ID: {id_}, skip this case but save the iteration process.")
                else:
                    print(f"\n================== Final Output ==================\n")
                    print(f"ID: {id_}\nQuestion: \n{question}\n\nLabel: \n{item['answer']}\n\nFinal Output: \n{final_answer}\n")
                    print('-------------------------------------------')
                
                
                writer.write({
                    'id': id_,
                    'query': question,
                    'label': item['answer'],
                    'final_answer': final_answer,
                    'iteration': iteration,
                    'total_retrieval_results': total_retrieval_results,
                    'supervisor_records': supervisor_records,
                    'extracted_facts': extract_facts,
                    'fact_dict': fact_dict,
                    'reasoner_records': reasoner_records,
                    'total_tokens': total_tokens,
                    'total_time': total_time,
                    'note': note,
                })
        ### 恢复标准输出
        sys.stdout.close()
        sys.stdout = sys.__stdout__
