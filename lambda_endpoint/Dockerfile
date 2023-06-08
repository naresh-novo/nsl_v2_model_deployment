FROM public.ecr.aws/lambda/python:3.9

WORKDIR .

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY scoring_endpoint.py .
COPY nsql_scoring_v2.json .

CMD ["nsl_v2_scoring_endpoint.lambda_handler"]