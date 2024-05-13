import functions_framework
from flask import jsonify
import traceback
from Normalizer import SkillNormalizerOld, JobNormalizer, LanguageNormalizer, \
                       SkillNormalizerNew, JobNormalizerWithHierarchy, CourseSkillNormalizer
from multiprocessing.pool import ThreadPool

ml_models = {}
ml_models["skill"] = SkillNormalizerOld()
ml_models["job"] = JobNormalizer()
ml_models["language"] = LanguageNormalizer()
ml_models["skill_new"] = SkillNormalizerNew()
ml_models["job_hierarchy"] = JobNormalizerWithHierarchy()
ml_models["course_skill"] = CourseSkillNormalizer()
def paral_proc(x):
  raw_string = x[0]
  norm_type = x[1]
  norm, add_info, notes, norm_info = ml_models[norm_type].normalize(raw_string)
  return {"normalized_string": norm, "additional_info": add_info, "notes": notes, "normalization_info": norm_info}

@functions_framework.http
def hello_world(request):
    try:
        # print(request)
        request_json = request.get_json()
        batch = request_json['calls']
        print(f'testing: {len(batch)}')
        norm_type = request_json['userDefinedContext']['norm_type']

        batch = [(x[0],norm_type) for x in batch]
        # print(batch)
        # Create a pool of workers
        with ThreadPool(100) as p:
          results = p.map(paral_proc, batch)

        # print(results)
        return_json = jsonify( { "replies":  results } )
        # print(return_json)
        return return_json

    except Exception as e:
        print(f'REQUEST: {request}')
        print(f'REQUEST: {request.get_json()}')
        print(traceback.format_exc())
        return jsonify( { "errorMessage": str(e) } ), 400
