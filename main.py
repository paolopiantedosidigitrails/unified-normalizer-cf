import functions_framework
#from multiprocessing.pool import ThreadPool
from flask import jsonify
import traceback
from Normalizer import SkillNormalizerOld, JobNormalizer, LanguageNormalizer, \
                       SkillNormalizerNew, JobNormalizerWithHierarchy, CourseSkillNormalizer

ml_models = {}
ml_models["skill"] = SkillNormalizerOld()
ml_models["job"] = JobNormalizer()
ml_models["language"] = LanguageNormalizer()
ml_models["skill_new"] = SkillNormalizerNew()
ml_models["job_hierarchy"] = JobNormalizerWithHierarchy()
ml_models["course_skill"] = CourseSkillNormalizer()
    
@functions_framework.http
def hello_world(request):
    try:
        # print(request)
        request_json = request.get_json()
        batch = request_json['calls']
        norm_type = request_json['userDefinedContext']['norm_type']
        # print(batch)
        # Create a pool of workers
        results = []
        for single_info in batch:
            raw_string = single_info[0]
            norm, add_info, notes, norm_info = ml_models[norm_type].normalize(raw_string)
            results.append({"normalized_string": norm, "additional_info": add_info, "notes": notes, "normalization_info": norm_info})

        # print(results)
        return_json = jsonify( { "replies":  results } )
        # print(return_json)
        return return_json

    except Exception as e:
        print(f'REQUEST: {request}')
        print(f'REQUEST: {request.get_json()}')
        print(traceback.format_exc())
        return jsonify( { "errorMessage": str(e) } ), 400
