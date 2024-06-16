from model.llava.constants import DEFAULT_IMAGE_TOKEN


SHORT_QUESTION_TEMPLATE = [
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you confirm if {class_name} is present in this image? If it is, kindly provide the segmentation map. If it isn't, please state so and, if appropriate, mention any objects that might be mistaken for {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is there a {class_name} visible in this image? If yes, I'd like to see the segmentation map. If not, please deny its presence and, as needed, note any similar items present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, do you detect the presence of {class_name}? Provide a segmentation map if it's there. If it's absent, please clarify and, if relevant, list any similar-looking objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please check for {class_name} in the image. Share a segmentation map if it exists. Otherwise, confirm its absence and, where applicable, identify objects resembling {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "I need to know if {class_name} is part of this image. If yes, please show the segmentation. If not, deny and, if it seems suitable, suggest any items that could be confused with {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Could you inspect this image for {class_name}? If found, supply the segmentation map. If not, refute its presence and, if it makes sense, highlight any similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Examine the image for the occurrence of {class_name}. If it's there, provide segmentation. If not, deny and, if it's fitting, note any objects that may resemble {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is {class_name} featured in this image? If so, a segmentation map is needed. If not, kindly refuse and, if deemed appropriate, mention any look-alike objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please verify the existence of {class_name} in this image. Present a segmentation map if applicable. Otherwise, refute and, where relevant, indicate any potentially confusing objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Assess this image for {class_name}. If present, I'd like the segmentation map. If absent, please deny and, if necessary, list any objects that might be mistaken for {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Check if {class_name} is in the image. Provide a segmentation map if it is. If it isn't, deny its presence and, where practical, suggest similar-looking items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Does the image include {class_name}? If yes, display the segmentation map. If no, reject and, if it fits, identify any objects that could be misinterpreted as {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Confirm the presence of {class_name} in the image. If it exists, show the segmentation. Otherwise, deny and, as applicable, specify objects that resemble {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you spot {class_name} in this image? If it's there, please provide the segmentation map. If not, deny and, if you see fit, mention any similar items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please identify if {class_name} is in this image. If you find it, offer the segmentation map. If you don't, refute and, if it seems right, point out any objects that might look like {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Examine this image for the presence of {class_name}. If detected, supply a segmentation map. If not, deny its presence and, when suitable, highlight any objects that could be mistaken for {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, is {class_name} present? If yes, please provide the segmentation. If not, state so and, as you deem fit, list any similar objects that are present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Review the image for {class_name}. If it’s there, I need the segmentation map. If it’s not, please confirm its absence and, if it appears relevant, note any similar items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is {class_name} part of this image's composition? If so, a segmentation map is required. If not, please deny and, if appropriate, mention any look-alike objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Investigate if {class_name} exists in this image. Provide a segmentation map if it does. If it doesn't, refute its presence and, if you think it necessary, suggest any resembling objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please scrutinize this image for {class_name}. If it's present, display the segmentation map. If not, decline and, where it makes sense, suggest any objects that might resemble {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Does this image contain {class_name}? If it does, please generate a segmentation map. If it doesn't, refute and, if relevant, list any similar objects present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "I'm interested in knowing if {class_name} is part of this image. If it is, provide the segmentation map. If it isn't, deny and, where applicable, identify any potentially confusing objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Could you check this image for the existence of {class_name}? If found, show the segmentation map. If not found, reject and, if suitable, point out similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, can you verify the presence of {class_name}? If yes, provide the segmentation. If no, deny and, if it seems appropriate, note any look-alike items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is {class_name} a feature in this image? If so, I need the segmentation map. If not, please clarify and, if fitting, highlight any objects that could be mistaken for {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Check for the presence of {class_name} in this image. If present, please produce a segmentation map. If absent, refute and, as needed, mention any similar-looking objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please inspect this image for {class_name}. If it's there, share the segmentation map. If it's not, deny its existence and, if you see fit, list any objects that resemble {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you detect {class_name} in this image? If present, I'd like to see the segmentation map. If absent, please reject and, if applicable, suggest similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Assess the image for the occurrence of {class_name}. If it's detected, provide the segmentation map. If not, deny and, if relevant, point out any objects that may be confused with {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is there any sign of {class_name} in this image? If yes, please produce the segmentation map. If no, refuse and, if appropriate, list any objects that might look like {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Confirm whether {class_name} appears in this image. If it does, share the segmentation. If not, deny and, where it makes sense, specify similar items present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Examine for {class_name} in this image. If you find it, please provide the segmentation map. If not, deny and, if fitting, point out any similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you spot any {class_name} in this image? If it's there, generate the segmentation map. If not, refuse and, if necessary, mention any items that could be mistaken for {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please look for {class_name} in this image. If present, output the segmentation map. If absent, deny and, if suitable, highlight any similar-looking objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Investigate this image for {class_name}. If it's visible, provide a segmentation map. If it's not, refute and, as applicable, list any objects that might resemble {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In the image, is {class_name} evident? If so, present the segmentation map. If not, please deny and, if it seems relevant, note any items that may look similar.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Look into this image for {class_name}. If it exists, I need the segmentation map. If it doesn't, confirm its absence and, if deemed necessary, note similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Does the image showcase {class_name}? If it does, kindly produce the segmentation map. If it doesn't, reject and, if it's appropriate, identify any resembling objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Examine the image to ascertain if {class_name} is included. If yes, provide a segmentation map. If no, deny and, if applicable, point out any objects that could be incorrectly identified as {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please analyze this image for any trace of {class_name}. If detected, provide the segmentation map. If not, deny and, if relevant, mention any objects that could be confused with {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you determine if {class_name} is depicted in this image? If yes, a segmentation map is needed. If no, please reject and, if it makes sense, list any objects that resemble {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Inspect the image to see if {class_name} is included. If it is, kindly provide the segmentation map. If not, refuse and, if appropriate, suggest similar-looking items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is {class_name} present in this image? If you find it, please show the segmentation map. If absent, deny and, if suitable, point out any objects that might be mistaken for {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Review this image for the presence of {class_name}. If present, offer the segmentation map. If not, deny and, if it seems fitting, identify any similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, can you locate {class_name}? If yes, provide the segmentation. If no, reject and, if it's relevant, note any items that may look like {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Examine this image for any signs of {class_name}. If it's there, I need the segmentation map. If it's not, please clarify and, if necessary, highlight any similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you confirm the existence of {class_name} in this image? If it’s there, provide a segmentation map. If not, deny and, if you see fit, list any objects resembling {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Check if this image includes {class_name}. If it does, show the segmentation map. If it doesn't, refuse and, if relevant, suggest any similar-looking objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please scan the image for {class_name}. If found, present the segmentation map. If not, deny its presence and, if it makes sense, identify any similar items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Assess whether {class_name} is visible in this image. If so, I require the segmentation map. If not, refute and, if fitting, mention any look-alike objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Examine the image to verify if {class_name} is present. If it is, provide the segmentation. If not, deny and, if it seems appropriate, note any objects that might be confused with {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is there an occurrence of {class_name} in this image? If yes, display the segmentation map. If no, reject and identify any resembling objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please check this image for {class_name}. If it’s present, supply the segmentation map. If it’s absent, refuse and point out similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Could you search the image for {class_name}? If you detect it, provide a segmentation map. If not, deny and suggest any objects that could be mistaken for {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, is there any evidence of {class_name}? If so, please produce the segmentation map. If not, decline and list any objects that may look similar.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Review the image for any sign of {class_name}. If it exists, I need the segmentation map. If it doesn't, confirm its absence and mention similar items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Does this image feature {class_name}? If it does, kindly generate the segmentation map. If it doesn't, refute and identify any resembling objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Inspect the image to establish if {class_name} is a part of it. If yes, provide a segmentation map. If no, deny and point out any objects that could be incorrectly identified as {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you verify whether {class_name} appears in this image? If present, display the segmentation. If absent, refuse and list any similar-looking objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Investigate this image and determine if {class_name} is featured. If so, provide the segmentation map. If not, deny its presence and suggest any similar-looking objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please analyze the image to find out if {class_name} is included. If it is, show the segmentation map. If it's not, clarify and note any objects that resemble {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, is there any indication of {class_name}? If found, provide the segmentation map. If not, reject and mention similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Check this image for {class_name}. If present, I'd like the segmentation map. If absent, please deny and highlight any objects that might be confused with {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you assess this image for the presence of {class_name}? If detected, supply the segmentation map. If not, refuse and list any look-alike objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is {class_name} identifiable in this image? If yes, display the segmentation map. If no, deny and point out any objects that could be mistaken for {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please scrutinize this image for any signs of {class_name}. If it exists, present the segmentation map. If not, refute and suggest any similar items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Examine the image for {class_name}. If you find it, provide the segmentation map. If it's not there, deny and indicate any objects that may resemble {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, could you check for the occurrence of {class_name}? If it's present, show the segmentation map. If not, decline and list similar-looking objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Search the image for {class_name}. If it's included, I need the segmentation map. If it's missing, please confirm its absence and note any similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is {class_name} part of this image's content? If so, produce the segmentation map. If not, deny and mention any items that might look like {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Check if {class_name} appears in this image. If yes, provide the segmentation map. If no, refute and highlight any similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, do you find any trace of {class_name}? If it's there, offer the segmentation. If not, reject and identify any resembling items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please confirm whether {class_name} is evident in this image. If it is, I'd like the segmentation map. If it isn't, deny and point out similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Examine for the presence of {class_name} in this image. If you detect it, provide a segmentation map. If not, refuse and list any objects that could be confused with {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is there any sign of {class_name} in this image? If yes, show the segmentation map. If not, please deny and suggest any objects that resemble {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you identify {class_name} in this image? If present, offer the segmentation map. If absent, refute and mention any look-alike objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Review this image for any indications of {class_name}. If found, provide the segmentation map. If not, deny and note any similar-looking items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please check the image for the occurrence of {class_name}. If it's there, I need the segmentation map. If it's not, clarify and highlight any objects that could be mistaken for {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, is {class_name} visible? If so, produce the segmentation map. If not, reject and list any items that may look like {class_name}.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is {class_name} in this image? If yes, please provide the segmentation map. If no, explicitly reject and optionally suggest similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you confirm if {class_name} is present in the image? If present, show the segmentation map. If absent, clearly deny and optionally list any similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please check for {class_name} in this image. If found, provide the segmentation map. If not found, explicitly state its absence, and optionally, mention similar items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Examine the image for {class_name}. If it’s there, provide the segmentation map. If it’s not, clearly deny presence and optionally note similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, is {class_name} present? If so, display the segmentation map. If not, directly deny and optionally identify any resembling objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is there any sign of {class_name} in this image? If yes, provide the segmentation map. If no, explicitly refuse and optionally list objects that might resemble it.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please verify if this image contains {class_name}. If it does, show the segmentation map. If it doesn’t, clearly state its absence and optionally suggest similar items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Inspect the image for any trace of {class_name}. If detected, supply the segmentation map. If not, outright deny and optionally mention look-alike objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you find {class_name} in this image? If present, provide the segmentation map. If absent, explicitly reject and optionally point out similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Examine the image for the occurrence of {class_name}. If present, offer the segmentation map. If not, directly deny and optionally identify similar-looking objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Check this image for the presence of {class_name}. If it’s there, present the segmentation map. If not, give a clear rejection and optionally suggest any similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Does this image feature {class_name}? If it does, provide the segmentation map. If not, clearly state the absence and optionally note any objects that could be confused with it.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is {class_name} detectable in this image? If yes, show the segmentation map. If no, explicitly deny and optionally highlight similar items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please search for {class_name} in the image. If found, supply the segmentation map. If not, openly refute and optionally list any resembling objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, can you locate {class_name}? If found, provide the segmentation. If not, clearly deny and optionally mention any similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Review the image for {class_name}. If it’s visible, offer the segmentation map. If not, outright reject and optionally point out similar-looking items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Is there evidence of {class_name} in this image? If so, provide the segmentation. If not, directly deny and optionally identify similar objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you identify {class_name} in this image? If present, show the segmentation map. If absent, directly refuse and optionally suggest any look-alike objects.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please analyze the image for {class_name}. If it’s there, present the segmentation map. If not, clearly state its absence and optionally note any similar items.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Examine the image to confirm if {class_name} is included. If yes, offer the segmentation. If no, give a clear denial and optionally list any resembling items.",
]

LONG_QUESTION_TEMPLATE = [
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Could you provide a segmentation map for the indicated area if it's present?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If you find the indicated area, can you show the segmentation map?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Can you generate a segmentation map for the area mentioned in the image?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Please confirm the presence of the indicated area and provide a segmentation map if it's there.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If this description matches the image, could you provide a segmentation map for the specified area?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Check for the described area in the image and segment it if you find it.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If the indicated area is present, I need a segmentation map for it.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Analyze the image for the area mentioned and display the segmentation if it's located.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If the area described is found in the image, please provide a segmentation map.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Examine the image for the specified area and segment it if it's present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Can you provide a segmentation map for the described area in the image?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If the description matches an area in the image, please provide its segmentation.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Analyze the image for the mentioned area. If it exists, segment this part.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Observe the described area in the image and provide a segmentation map if it matches.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Confirm and provide segmentation for the indicated area in the image if it's present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If you find the described area in the image, provide a segmentation map.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Check the image for the specified area and provide a segmentation map if it's there.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Segment the described area in the image if it is indicated.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Inspect the image for the mentioned area and provide a segmentation map if visible.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Provide a segmentation map for the described area in the image if it matches.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Show the segmentation for the specified area in the image if it's present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Segment the area mentioned in the image if the description is correct.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If the area described is present in the image, provide a segmentation map.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Confirm the presence of the specified area in the image with a segmentation map.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Provide a segmentation map for the indicated area in the image if it's present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Verify the described area in the image and provide a segmentation map if it's there.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Review the image for the mentioned area and segment it if the description matches.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If you find the described area in the image, provide a segmentation map.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} I'd like a segmentation map for the specified area in the image if it's present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If the image matches the description, please provide a segmentation map for the specified area.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Provide the segmentation for the specified area in the image if the description is accurate.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Segment the mentioned area in the image if it is present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Please provide a segmentation map for the described area in the image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Confirm the presence of the specified area in the image and provide segmentation if present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If the described area is visible in the image, can you segment this part?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If the specified area is present in the image, provide a segmentation map.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Display the segmentation for the area mentioned in the image if located.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Segment the described area in the image if found.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If the specified area is present in the image, can you segment it?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Provide segmentation for the described area in the image if it's present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If the description matches the image, provide the segmentation for the specified area.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Segment the indicated area in the image if it exists.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If the image matches the described area, segment this part.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Confirm and provide segmentation for the specified area in the picture if present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If you find the specified area in the image, provide a segmentation map for it.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Check the image for the described area and provide a segmentation map if present.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Segment the indicated area in the image if it's visible.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} Inspect the image for the specified area and provide a segmentation map if it's there.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "{sent} If the image has the described area, please provide a segmentation map for it.",
]

NEG_ANSWER_TEMPLATE = [
    "No presence of {class_name} detected in this image.",
    "Unable to locate {class_name} here.",
    "Sorry, {class_name} is not found in this image.",
    "I don't see {class_name} in the image.",
    "No, there's no {class_name} in this picture.",
    "The image does not contain {class_name}.",
    "Regrettably, {class_name} is absent from the image.",
    "No sign of {class_name} in the image.",
    "No, {class_name} isn't visible here.",
    "I can't find {class_name} in this image.",
    "{class_name} is not present in the image.",
    "There's no trace of {class_name} here.",
    "Unfortunately, the image lacks {class_name}.",
    "No, the image doesn't include {class_name}.",
    "Sorry, but {class_name} isn't in this image.",
    "I cannot detect {class_name} in the image.",
    "{class_name} doesn't seem to be in the image.",
    "After checking, {class_name} isn't present.",
    "The image doesn't feature {class_name}.",
    "There's no {class_name} in this particular image.",
    "This image appears to be void of any {class_name}.",
    "{class_name} seems to be missing from this image.",
    "I've scanned thoroughly, but {class_name} is not here.",
    "Alas, this image does not showcase {class_name}.",
    "It appears that {class_name} is not part of this scene.",
    "Scanning complete: no {class_name} detected.",
    "No evidence of {class_name} in this visual.",
    "This depiction lacks a {class_name}.",
    "Absent: {class_name} in this imagery.",
    "I'm not able to spot {class_name} here.",
    "The search yielded no results for {class_name}.",
    "No {class_name} in sight in this frame.",
    "This image is devoid of any {class_name}.",
    "No visual confirmation of {class_name} found.",
    "A thorough look reveals no {class_name}.",
    "This image seems to be missing a {class_name}.",
    "No, this picture doesn't seem to have {class_name}.",
    "I can confirm that {class_name} is not in the image.",
    "A detailed inspection shows no {class_name}.",
    "Regret to inform that {class_name} is not depicted here.",
    "On closer observation, no {class_name} is present.",
    "No instance of {class_name} can be seen here.",
    "I'm unable to confirm the presence of {class_name}.",
    "This visual representation lacks {class_name}.",
    "After a careful look, {class_name} is missing.",
    "No, this depiction is devoid of {class_name}.",
    "This image fails to display any {class_name}.",
    "Upon review, {class_name} is notably absent.",
    "A comprehensive scan finds no {class_name}.",
    "There seems to be an absence of {class_name} here.",
    "No, there's no {class_name} in this image.",
    "Unable to locate {class_name} here.",
    "This image doesn't seem to feature any {class_name}.",
    "I don't see it in the image.",
    "No, this picture doesn't have it.",
    "The image does not contain {class_name}.",
    "It's absent from this image.",
    "No sign of {class_name} in the image.",
    "I'm not able to spot it here.",
    "No, it's not present.",
    "{class_name} isn't visible in this image.",
    "There's no trace of it.",
    "Unfortunately, the image lacks {class_name}.",
    "No, the image doesn't include it.",
    "Sorry, but it isn't in this image.",
    "I cannot detect {class_name} in the image.",
    "Doesn't seem to be in the image.",
    "After checking, it's not present.",
    "The image doesn't feature {class_name}.",
    "No, there's nothing like that here.",
    "This image appears to be void of any {class_name}.",
    "Seems to be missing from this image.",
    "I've scanned thoroughly, but it's not here.",
    "Alas, this image does not showcase {class_name}.",
    "It appears that it's not part of this scene.",
    "Scanning complete: no {class_name} detected.",
    "No evidence of it in this visual.",
    "This depiction lacks a {class_name}.",
    "Absent: {class_name} in this imagery.",
    "I'm not detecting it in this visual.",
    "The search yielded no results for {class_name}.",
    "No {class_name} in sight in this frame.",
    "This image is devoid of any {class_name}.",
    "No visual confirmation of it found.",
    "A thorough look reveals no {class_name}.",
    "This image seems to be missing it.",
    "No, this picture doesn't seem to have {class_name}.",
    "I can confirm that it's not in the image.",
    "A detailed inspection shows no {class_name}.",
    "Regret to inform that it's not depicted here.",
    "On closer observation, no {class_name} is present.",
    "No instance of it can be seen here.",
    "I'm unable to confirm the presence of {class_name}.",
    "This visual representation lacks it.",
    "After a careful look, {class_name} is missing.",
    "No, this depiction is devoid of {class_name}.",
    "This image fails to display any {class_name}.",
    "Upon review, it is notably absent.",
    "A comprehensive scan finds no {class_name}.",
    "There seems to be an absence of it here.",
]

LONG_ANSWER_TEMPLATE = [
    "Confirmed, here's the segmentation: [SEG].",
    "Detection acknowledged: [SEG].",
    "Indeed, the object is present. Segmentation map: [SEG].",
    "Affirmative. Providing the segmentation: [SEG].",
    "Object found. Here's the segmentation: [SEG].",
    "It's there. Segmentation provided: [SEG].",
    "Yes, I see it. Segmentation map: [SEG].",
    "Detection successful. Segmentation: [SEG].",
    "Correct. It's in the image: [SEG].",
    "Found it! Segmentation map: [SEG].",
    "Yes, it is here. Segmentation: [SEG].",
    "Absolutely, it's visible. Segmentation: [SEG].",
    "Sure, it is present. Here's the segmentation: [SEG].",
    "Located. Segmentation map provided: [SEG].",
    "Indeed, that's correct. Segmentation: [SEG].",
    "Yes, included in the image. Segmentation: [SEG].",
    "Confirmed. Providing the segmentation: [SEG].",
    "It's a match. Segmentation: [SEG].",
    "Yes, part of this image. Segmentation: [SEG].",
    "Definitely in the image. Segmentation: [SEG].",
    "Segmentation coming up: [SEG].",
    "I can confirm its presence: [SEG].",
    "Positively identified. Here's the segmentation: [SEG].",
    "Definitely found. Segmentation map: [SEG].",
    "Yes, right here. Segmentation: [SEG].",
    "Spotted successfully. Segmentation map: [SEG].",
    "Visible and detected. Segmentation map: [SEG].",
    "Presence confirmed. Here's the segmentation: [SEG].",
    "Detected in the image. Segmentation: [SEG].",
    "Right there! Spotted. Segmentation map: [SEG].",
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

SHORT_ANSWER_TEMPLATE= [
    "Confirmed, here's the segmentation: [SEG].",
    "Yes, {class_name} is detected: [SEG].",
    "Indeed, {class_name} is present. Segmentation map: [SEG].",
    "Affirmative. Segmentation for {class_name}: [SEG].",
    "{class_name} found. Here's the segmentation: [SEG].",
    "It's there. Segmentation for {class_name}: [SEG].",
    "Yes, I see it. Segmentation map for {class_name}: [SEG].",
    "Detection successful. {class_name} segmentation: [SEG].",
    "Correct. {class_name} is in the image: [SEG].",
    "Found it! Segmentation map of {class_name}: [SEG].",
    "Yes, {class_name} is here. Segmentation: [SEG].",
    "Absolutely, {class_name} is visible. Segmentation: [SEG].",
    "Sure, {class_name} is present. Here's the segmentation: [SEG].",
    "{class_name} located. Segmentation map provided: [SEG].",
    "Indeed, that's {class_name}. Segmentation: [SEG].",
    "Yes, the image includes {class_name}. Segmentation: [SEG].",
    "Confirmed. Segmentation of {class_name}: [SEG].",
    "It's a match. {class_name} segmentation: [SEG].",
    "Yes, {class_name} is a part of this image. Segmentation: [SEG].",
    "{class_name} is definitely in the image. Segmentation: [SEG].",
    "Segmentation for {class_name} coming up: [SEG].",
    "I can confirm that {class_name} is indeed present: [SEG].",
    "Positive identification of {class_name}. Here's the segmentation: [SEG].",
    "Definitely, {class_name} is found. Segmentation map: [SEG].",
    "Yes, {class_name} is right here. Segmentation: [SEG].",
    "{class_name} is spotted. Segmentation map: [SEG].",
    "Found {class_name} successfully. Segmentation: [SEG].",
    "Visible and detected: {class_name}. Segmentation map: [SEG].",
    "{class_name} presence confirmed. Here's the segmentation: [SEG].",
    "Detected {class_name} in the image. Segmentation: [SEG].",
    "Right there! {class_name} spotted. Segmentation map: [SEG].",
    "Yes, {class_name} is definitely in the frame. Segmentation: [SEG].",
    "Got it, {class_name} located. Segmentation provided: [SEG].",
    "That's a positive for {class_name}. Segmentation map: [SEG].",
    "Indeed, we have {class_name} in sight. Segmentation: [SEG].",
    "Clear view of {class_name}. Here's the segmentation: [SEG].",
    "{class_name} is unmistakably present. Segmentation: [SEG].",
    "Identified {class_name} with clarity. Segmentation: [SEG].",
    "Segmentation ready for {class_name}: [SEG].",
    "Absolutely certain, that's {class_name}. Segmentation: [SEG].",
    "Yes, without a doubt, {class_name} is there. Segmentation: [SEG].",
    "Found what you're looking for - {class_name}: [SEG].",
    "There's no mistaking it, that's {class_name}. Segmentation: [SEG].",
    "{class_name} is clearly in the picture. Segmentation: [SEG].",
    "Undoubtedly, {class_name} is present. Segmentation map: [SEG].",
    "{class_name} has been successfully identified. Segmentation: [SEG].",
    "I've located {class_name} in the image. Segmentation: [SEG].",
    "{class_name} is right there, as you suspected. Segmentation: [SEG].",
    "The image definitely contains {class_name}. Segmentation: [SEG].",
    "After thorough analysis, {class_name} is found. Segmentation: [SEG].",
    "{class_name} is evident in the visual. Segmentation: [SEG].",
    "Confirmed, here's the segmentation: [SEG].",
    "Detection acknowledged: [SEG].",
    "Indeed, the object is present. Segmentation map: [SEG].",
    "Affirmative. Providing the segmentation: [SEG].",
    "Object found. Here's the segmentation: [SEG].",
    "It's there. Segmentation provided: [SEG].",
    "Yes, I see it. Segmentation map: [SEG].",
    "Detection successful. Segmentation: [SEG].",
    "Correct. It's in the image: [SEG].",
    "Found it! Segmentation map: [SEG].",
    "Yes, it is here. Segmentation: [SEG].",
    "Absolutely, it's visible. Segmentation: [SEG].",
    "Sure, it is present. Here's the segmentation: [SEG].",
    "Located. Segmentation map provided: [SEG].",
    "Indeed, that's correct. Segmentation: [SEG].",
    "Yes, included in the image. Segmentation: [SEG].",
    "Confirmed. Providing the segmentation: [SEG].",
    "It's a match. Segmentation: [SEG].",
    "Yes, part of this image. Segmentation: [SEG].",
    "Definitely in the image. Segmentation: [SEG].",
    "Segmentation coming up: [SEG].",
    "I can confirm its presence: [SEG].",
    "Positively identified. Here's the segmentation: [SEG].",
    "Definitely found. Segmentation map: [SEG].",
    "Yes, right here. Segmentation: [SEG].",
    "Spotted successfully. Segmentation map: [SEG].",
    "Visible and detected. Segmentation map: [SEG].",
    "Presence confirmed. Here's the segmentation: [SEG].",
    "Detected in the image. Segmentation: [SEG].",
    "Right there! Spotted. Segmentation map: [SEG].",
]

CORRECT_ANSWER_TEMPLATE= [
    "While the {class_name} is not present, the image does include {gt_name}.",
    "There's no {class_name} here, but I did find {gt_name} in the image.",
    "Sorry, I cannot locate {class_name}. However, {gt_name} is visible.",
    "The {class_name} is absent, but you might be interested in {gt_name} that's in the picture.",
    "No {class_name} detected, but the image prominently features {gt_name}.",
    "I couldn't spot {class_name}, but there's a clear presence of {gt_name}.",
    "The image doesn't have {class_name}, but it does contain {gt_name}.",
    "Unfortunately, {class_name} isn't in this image. However, {gt_name} is.",
    "No sign of {class_name}, but {gt_name} can be seen instead.",
    "{class_name} is missing, yet {gt_name} makes a noticeable appearance.",
    "Absent {class_name}, but the {gt_name} is quite prominent.",
    "While {class_name} isn't there, {gt_name} is definitely in the frame.",
    "I can't confirm {class_name}, but {gt_name} is certainly present.",
    "{class_name} is not in the shot, but take a look at {gt_name}.",
    "Not finding {class_name}, but you might find {gt_name} interesting.",
    "No trace of {class_name}, but don't miss {gt_name} in the same image.",
    "{class_name} isn't featured, but the picture does show {gt_name}.",
    "Can't validate {class_name}, but {gt_name} is clearly visible.",
    "The search for {class_name} turned up empty, however, {gt_name} is in sight.",
    "Although {class_name} is nowhere to be found, {gt_name} is definitely included.",
    "No appearance of {class_name}, but the image does capture {gt_name}.",
    "Missing {class_name}, yet the image reveals {gt_name}.",
    "The {class_name} isn't visible, but {gt_name} stands out.",
    "Can't spot {class_name}, but {gt_name} is a key element here.",
    "{class_name} is not present, but {gt_name} is quite evident.",
    "Though {class_name} is missing, {gt_name} is clearly depicted.",
    "No {class_name} in view, but do take note of {gt_name}.",
    "The {class_name} isn't here, but the image does feature {gt_name}.",
    "I don't see {class_name}, but {gt_name} is prominently displayed.",
    "While we're missing {class_name}, {gt_name} is clearly visible.",
    "No {class_name} found, but the image is rich with {gt_name}.",
    "The image lacks {class_name}, yet showcases {gt_name}.",
    "{class_name} is not detected, but there's a distinct {gt_name}.",
    "No evidence of {class_name}, but check out the {gt_name}.",
    "Absent {class_name}, yet there's a striking {gt_name}.",
    "I can't find {class_name}, but the image includes {gt_name}.",
    "{class_name} isn't there, but {gt_name} is a notable feature.",
    "No {class_name} to be seen, however, {gt_name} is present.",
    "The {class_name} doesn't feature, but you can see {gt_name}.",
    "Can't locate {class_name}, but the image highlights {gt_name}.",
    "There's an absence of {class_name}, with {gt_name} in focus.",
    "No {class_name} here, but the image spotlights {gt_name}.",
    "{class_name} is not apparent, but the image emphasizes {gt_name}.",
    "The {class_name} isn't part of the scene, but {gt_name} is evident.",
    "Can't discern {class_name}, yet {gt_name} is prominently featured.",
    "While {class_name} is missing, the image is rich with {gt_name}.",
    "No {class_name}, but take a look at the {gt_name} in there.",
    "{class_name} isn't in the frame, but {gt_name} takes center stage.",
    "Absent {class_name}, however, {gt_name} plays a major role.",
    "I don't find {class_name}, but the image clearly shows {gt_name}.",
    "There's no {class_name}, but look at the {gt_name}.",
    "I can't see {class_name}, though there is {gt_name}.",
    "The {class_name} isn't here, but there's {gt_name} in the image.",
    "No {class_name} found, but you can see {gt_name}.",
    "{class_name} isn't in the picture, instead, it's {gt_name}.",
    "I don't see {class_name}, but {gt_name} is present.",
    "The {class_name} is missing, but you'll notice {gt_name}.",
    "Can't find {class_name}, but there's clearly {gt_name}.",
    "While there's no {class_name}, the image has {gt_name}.",
    "No sign of {class_name}, but the {gt_name} is there.",
    "Absent {class_name}, but you'll find {gt_name}.",
    "I couldn't spot {class_name}, but did see {gt_name}.",
    "The {class_name} isn't visible, but {gt_name} is.",
    "There's no {class_name}, however, {gt_name} is included.",
    "{class_name} isn't in this one, but there is {gt_name}.",
    "The image lacks {class_name}, but not {gt_name}.",
    "No {class_name} here, yet there's {gt_name}.",
    "Couldn't find {class_name}, but {gt_name} is right there.",
    "The {class_name} is absent, but {gt_name} is visible.",
    "No {class_name}, but the image does have {gt_name}.",
    "I don't find {class_name}, but I do see {gt_name}.",
    "While {class_name} is missing, you can see {gt_name}.",
    "No {class_name}, instead, there's {gt_name}.",
    "Can't spot {class_name}, but {gt_name} is in the frame.",
    "The {class_name} isn't here, but check out {gt_name}.",
    "Missing {class_name}, but {gt_name} is present.",
    "No trace of {class_name}, yet {gt_name} is there.",
    "The search for {class_name} turned up nothing, but there's {gt_name}.",
    "Although {class_name} isn't found, {gt_name} is.",
    "Can't validate {class_name}, but {gt_name} is visible.",
    "The {class_name} isn't part of this, but there is {gt_name}.",
    "While no {class_name}, the image features {gt_name}.",
    "I can't confirm {class_name}, but there's {gt_name}.",
    "{class_name} isn't featured, yet {gt_name} is.",
    "No {class_name} in sight, but {gt_name} is noticeable.",
    "Can't see {class_name}, but {gt_name} is clear.",
    "The {class_name} doesn't appear, but you can see {gt_name}.",
    "No {class_name}, though {gt_name} is there.",
    "{class_name} is not detected, but {gt_name} is.",
    "I'm not finding {class_name}, but there's {gt_name}.",
    "There's no {class_name}, but you might notice {gt_name}.",
    "The {class_name} isn't in the shot, but {gt_name} is.",
    "Can't locate {class_name}, but the {gt_name} is evident.",
    "No {class_name} to be seen, but there is {gt_name}.",
    "Absent {class_name}, however, you can spot {gt_name}.",
    "I don't detect {class_name}, but {gt_name} is there.",
    "No {class_name} here, but the image includes {gt_name}.",
    "While {class_name} is not present, {gt_name} is.",
    "The {class_name} isn't in this image, but {gt_name} is.",
]

EXPLANATORY_QUESTION_TEMPLATE = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]