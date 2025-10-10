import re


def parse_tom_story_to_code(story_text, include_comments=True):
    code_lines = []    
    code_lines.append("events = []")
    code_lines.append("observers = Set()")
    
    lines = [line.strip() for line in story_text.strip().split('\n') if line.strip()]
    
    for line_num, line in enumerate(lines, 1):
        # Add comment for original story line (if enabled)
        if include_comments:
            code_lines.append(f"# Line {line_num}: {line}")
        
        # Pattern 1: Multiple people entered
        if match := re.search(r'(.+?) entered the (.+?)\.', line):
            names, location = [match.group(i) for i in range(1, 3)]
            
            # Parse comma-separated names with "and"
            name_list = re.split(r',\s*and\s+|,\s*|\s+and\s+', names)
            name_list = [name.strip() for name in name_list if name.strip()]
            
            for name in name_list:
                code_lines.append(f"observers.add(Observer('{name}'))")
        
        # Pattern 2: Someone exited
        elif match := re.search(r'(\w+) exited the (.+?)\.', line):
            name = match.group(1)
            code_lines.append(f"observers.remove_by_name('{name}')")
        
        # Pattern 3: Someone moved something
        elif match := re.search(r'(\w+) moved the (\w+) to the (.+?)\.', line):
            agent, object_name, container = [match.group(i) for i in range(1, 4)]            
            code_lines.append(f"events.append(MoveEvent(observers=deepcopy(observers), agent='{agent}', object='{object_name}', container='{container}'))")
        
        # Pattern 4: Initial object location
        elif match := re.search(r'The (\w+) is in the (.+?)\.', line):
            object_name, container = [match.group(i) for i in range(1, 3)]
            code_lines.append(f"events.append(MoveEvent(observers=deepcopy(observers), agent='System', object='{object_name}', container='{container}'))")
        
        # Pattern 5: Other events (likes, lost, saw) - use OtherEvent class
        elif any(keyword in line.lower() for keyword in ['likes', 'lost', 'saw']):
            # Extract agent, verb, and object/target from the line
            if match := re.search(r'(\w+) (likes|lost|saw) (.+?)\.', line):
                agent = match.group(1)
                verb = match.group(2)
                target = match.group(3)
                code_lines.append(f"events.append(OtherEvent(observers=deepcopy(observers), agent='{agent}', object='{target}', verb='{verb}'))")
                if include_comments:
                    code_lines.append(f"# Other event: {agent} {verb} {target}")
            else:
                if include_comments:
                    code_lines.append(f"# Other event: {line} (could not parse)")
        
        else:
            if include_comments:
                code_lines.append(f"# Unrecognized pattern: {line}")
    
    return code_lines

def parse_tom_question_to_code(question, include_comments=True):
    question = question.strip()
    code_lines = []
    
    if include_comments:
        code_lines.append(f"# Question: {question}")
    
    # Pattern 1: Reality question - "Where is the X really?" or "Where is the X?"
    if match := re.search(r'Where is the (\w+)(?:\s+really)?\?', question, re.IGNORECASE):
        object_name = match.group(1)
        code_lines.append(f"safe_get([event for event in events if event.object == '{object_name}'], -1, 'container')")
    
    # Pattern 2: First-order belief - "Where does Y think the X is?"
    elif match := re.search(r'Where does (\w+) think the (\w+) is\?', question, re.IGNORECASE):
        observer, object_name = [match.group(i) for i in range(1, 3)]
        code_lines.append(f"safe_get([event for event in events if event.observers.has('{observer}', actual=True) and event.object == '{object_name}'], -1, 'container')")

    # Pattern 3: Second-order belief - "Where does A think B thinks the X is?"
    elif match := re.search(r'Where does (\w+) think (\w+) thinks the (\w+) is\?', question, re.IGNORECASE):
        observer1, observer2, object_name = [match.group(i) for i in range(1, 4)]
        code_lines.append(f"safe_get([event for event in events if event.observers.has('{observer1}', actual=True) and event.observers.has('{observer2}', perceived_by_others=True) and event.object == '{object_name}'], -1, 'container')")
    
    else:
        if include_comments:
            code_lines.append(f"# Unrecognized question pattern: {question}")
        code_lines.append("'Unknown question pattern'")
    
    return '\n'.join(code_lines)

def generate_question_code(object_name='pear'):
    """Generate code strings for common ToM questions"""
    
    question_code = []
    question_code.append("# Theory of Mind Questions")
    question_code.append("")
    
    # Real location
    question_code.append(f"# Where is the {object_name} really?")
    question_code.append(f"real_location = safe_get([event for event in events if event.object == '{object_name}'], -1, 'container')")
    question_code.append("print(f'Real location: {real_location}')")
    question_code.append("")
    
    # First-order beliefs template
    question_code.append("# First-order beliefs: Where does X think the object is?")
    question_code.append("# Replace 'OBSERVER_NAME' with actual observer name")
    question_code.append(f"# belief = safe_get([event for event in events if event.observers.has('OBSERVER_NAME', actual=True) and event.object == '{object_name}'], -1, 'container')")
    question_code.append("# print(f'OBSERVER_NAME thinks the object is in: {belief}')")
    question_code.append("")
    
    # Second-order beliefs template  
    question_code.append("# Second-order beliefs: Where does X think Y thinks the object is?")
    question_code.append("# Replace 'OBS1' and 'OBS2' with actual observer names")
    question_code.append(f"# shared_belief = safe_get([event for event in events if event.observers.has('OBS1', actual=True) and event.observers.has('OBS2', perceived_by_others=True) and event.object == '{object_name}'], -1, 'container')")
    question_code.append("# print(f'OBS1 thinks OBS2 thinks the object is in: {shared_belief}')")
    
    return question_code