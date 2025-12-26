"""
BioASQ prompt generation module.
Supports summary, list, factoid, yesno.
"""
import re


def format_triplets_bioasq(scored_triplets, thres=0.0):
    if not scored_triplets:
        return "No relevant knowledge triples found."
    
    # Filter low-score triples.
    filtered = [t for t in scored_triplets if len(t) >= 4 and t[3] >= thres]
    
    if not filtered:
        return "No relevant knowledge triples found (all below threshold)."
    
    # Format as text.
    lines = []
    for i, (h, r, t, score) in enumerate(filtered[:100], 1):  # Limit to 100.
        lines.append(f"{i}. {h} → {r} → {t} (score: {score:.3f})")
    
    return "\n".join(lines)


def format_snippets_bioasq(snippets):
    if not snippets:
        return "No text snippets available."
    
    lines = []
    for i, snippet in enumerate(snippets[:5], 1):  # Limit to 5.
        # Truncate length.
        text = snippet[:500] + "..." if len(snippet) > 500 else snippet
        lines.append(f"[Snippet {i}] {text}")
    
    return "\n\n".join(lines)


def get_bioasq_system_prompt(question_type):
    base_prompt = """You are a biomedical expert assistant. You will be given a biomedical question along with:
1. Relevant knowledge graph triples (entities and their relationships)
2. Text snippets from scientific literature

Use this information to answer the question accurately."""
    
    type_specific = {
        'summary': "\n\nFor this SUMMARY question, provide a comprehensive answer in 2-3 sentences.",
        'list': "\n\nFor this LIST question, provide your answer as a comma-separated list of entities.",
        'factoid': "\n\nFor this FACTOID question, provide a single, concise answer (entity, date, number, etc.).",
        'yesno': "\n\nFor this YES/NO question, first answer 'Yes' or 'No', then provide a brief explanation."
    }
    
    return base_prompt + type_specific.get(question_type, '')


def get_bioasq_user_prompt(question, question_type, scored_triplets, snippets, thres=0.0):
    # Format knowledge graph triples.
    kg_text = format_triplets_bioasq(scored_triplets, thres)

    # Format literature snippets.
    snippets_text = format_snippets_bioasq(snippets)

    # Assemble prompt.
    prompt = f"""Question: {question}

===== Knowledge Graph Evidence =====
{kg_text}

===== Literature Evidence =====
{snippets_text}

===== Task =====
"""
    
    # Add question-type specific instructions.
    if question_type == 'summary':
        prompt += "Provide a comprehensive summary answer (2-3 sentences) based on the evidence above."
    elif question_type == 'list':
        prompt += "List all relevant entities as a comma-separated list."
    elif question_type == 'factoid':
        prompt += "Provide the most accurate single answer."
    elif question_type == 'yesno':
        prompt += "Answer 'Yes' or 'No', followed by a brief explanation."
    else:
        prompt += "Provide a clear and accurate answer based on the evidence above."
    
    prompt += "\n\nAnswer:"
    
    return prompt


def get_prompts_for_data_bioasq(data, prompt_mode='scored_100', thres=0.0):
    print(f"Generating BioASQ prompts (mode={prompt_mode}, thres={thres})...")
    
    for idx, item in enumerate(data):
        question = item['question']
        question_type = item.get('question_type', 'summary')
        scored_triplets = item.get('scored_triplets', [])
        snippets = item.get('snippets', [])
        
        # Build system and user prompts.
        system_prompt = get_bioasq_system_prompt(question_type)
        user_prompt = get_bioasq_user_prompt(
            question, question_type, scored_triplets, snippets, thres
        )
        
        # Store fields expected by llm_utils.py.
        data[idx]['sys_query'] = system_prompt
        data[idx]['user_query'] = user_prompt
        data[idx]['cot_query'] = "Please provide your final answer."

        # Keep compatibility fields (optional).
        data[idx]['system_prompt'] = system_prompt
        data[idx]['user_prompt'] = user_prompt
        data[idx]['full_prompt'] = f"{system_prompt}\n\n{user_prompt}"
    
    print(f"Generated prompts for {len(data)} questions")
    return data


def unique_preserve_order(seq):
    """Deduplicate while preserving order."""
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]
