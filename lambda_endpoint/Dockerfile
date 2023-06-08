FROM public.ecr.aws/lambda/python:3.9

WORKDIR .

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY nsl_v2_scoring_endpoint.py .
COPY nsql_model_v2.json .

CMD ["nsl_v2_scoring_endpoint.lambda_handler"]