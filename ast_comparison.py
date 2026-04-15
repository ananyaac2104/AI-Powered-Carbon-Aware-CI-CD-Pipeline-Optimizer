import boto3
import json
from botocore.exceptions import ClientError
from deepdiff import DeepDiff

class ASTStorageManager:
    def __init__(self, bucket_name):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name

    def process_module_ast(self, module_name, current_ast_dict):
        """
        Checks if module exists. If new, stores to S3. 
        If exists, downloads and compares structurally.
        """
        # Define the S3 path for this module's AST
        object_key = f"ast_artifacts/{module_name}_ast.json"

        try:
            # Attempt to fetch the previous AST from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_key)
            previous_ast_json = response['Body'].read().decode('utf-8')
            previous_ast_dict = json.loads(previous_ast_json)

            print(f"Module '{module_name}' found in S3. Comparing structural ASTs...")
            return self._compare_asts(previous_ast_dict, current_ast_dict)

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                # The file doesn't exist, meaning this is a NEW module
                print(f"Module '{module_name}' is new. Uploading AST to S3...")
                self._store_ast(object_key, current_ast_dict)
                return {"status": "NEW_MODULE", "diff": None}
            else:
                # Re-raise if it's a permissions or network error
                raise e

    def _store_ast(self, object_key, ast_dict):
        """Helper to upload AST to S3"""
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=object_key,
            Body=json.dumps(ast_dict)
        )

    def _compare_asts(self, prev_ast, curr_ast):
        """
        Performs a pure structural comparison of the AST dictionaries.
        Ignores order in lists (like imports) to prevent false positives.
        """
        # DeepDiff will tell you exactly which nodes were added, removed, or changed
        diff = DeepDiff(prev_ast, curr_ast, ignore_order=True)
        
        if not diff:
            return {"status": "UNCHANGED", "diff": None}
        
        return {"status": "MODIFIED", "diff": diff}

# --- Example Usage ---
# manager = ASTStorageManager(bucket_name="my-ci-cd-artifacts")
# result = manager.process_module_ast("auth_handler", my_parsed_ast)