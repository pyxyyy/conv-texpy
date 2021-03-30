"""
Low-level utility functions to communicate with AMT.
"""
import json
import logging
from datetime import datetime
from typing import Optional, List
from xml.etree import ElementTree

import boto3
from boto.mturk.question import HTMLQuestion
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

DATETIME_FORMAT = "{:%Y-%m-%d %H:%M:%S}"


def get_client(use_prod: bool = False):
    """
    Get a AMT client client.
    """
    if use_prod:
        # NOTE: MTurk is curretly only offered in US-EAST-1, so we'll
        # set the region here.
        return boto3.client('mturk', region_name="us-east-1")
    else:
        return boto3.client('mturk', region_name="us-east-1", endpoint_url='https://mturk-requester-sandbox.us-east-1.amazonaws.com')


def get_account_balance(conn) -> float:
    response = conn.get_account_balance()
    return response['AvailableBalance']


# region: qualifications
def create_qualification(conn, name: str, keywords: str, description: str, auto_granted: bool = True,
                         auto_granted_value: int = 100) -> str:
    """
    Creates a new qualification for a task.

    @returns the qualification type id.
    """
    response = conn.create_qualification_type(
        Name=name,
        Keywords=keywords,
        Description=description,
        QualificationTypeStatus='Active',
        AutoGranted=auto_granted,
        AutoGrantedValue=auto_granted_value,
    )

    return response['QualificationType']['QualificationTypeId']


def get_qualification(conn, qual_id: str, worker_id: str) -> Optional[int]:
    """
    Gets the qualification score for a particular worker.
    """
    try:
        resp = conn.get_qualification_score(
            QualificationTypeId=qual_id,
            WorkerId=worker_id)
        return resp['Qualification']['IntegerValue']
    except ClientError:
        return None


def set_qualification(conn, qual_id: str, worker_id: str, value: int, *, reason: str = ""):
    """
    Update Qualification value for a user.
    """
    conn.associate_qualification_with_worker(
        QualificationTypeId=qual_id,
        WorkerId=worker_id,
        IntegerValue=value,
        SendNotification=False,
    )

    if reason:
        message_worker(conn, "Feedback on task", f"""\
We just wanted to share some feedback on your work: we have noted that
{reason}. We aren't rejecting your HIT, but have decreased your
qualification score to {value}. Contact us if you
would like more details.""", worker_id)


def update_qualifications(conn, qual_id: str, worker_id: str, update: int,
                          min_limit: int = 0, max_limit: int = 105, *, reason: str = "") -> int:
    """
    Update a worker's qualification. 

    @returns: the updated qualification value.
    """
    value = get_qualification(conn, qual_id, worker_id)
    value = min(max_limit, max(min_limit, update + value))
    set_qualification(conn, qual_id, worker_id, value)

    if reason and value < 0:
        message_worker(conn, "Feedback on task", f"""\
We just wanted to share some feedback on your work: we have noted that
{reason}. We aren't rejecting your HIT, but have decreased your
qualification score by {update}. It is currently {value} and we'll
automatically increase it when we approve your work. Contact us if you
would like more details.""", worker_id)

    return value


def prepare_qualification(qual: str) -> dict:
    """
    Prepares a qualification object from an expression string.
    Examples:
        location in US,CA,GB
        location not_in US-CA,US-MA
        hits_approved > 500
        percent_approved > 95

    :param qual: a qualification in format 'qual_id operator value'
    :return:
    """
    _QUAL_IDS = {
        "location": '00000000000000000071',
        "hits_approved": '00000000000000000040',
        "percent_approved": '000000000000000000L0'
    }
    _COMPARATORS = {
        ">": "GreaterThan",
        "=": "GreaterThan",
        "<": "LessThan",
        "in": "In",
        "not_in": "NotIn",
    }

    parts = qual.split(' ')
    if len(parts) != 3 or parts[1] not in _COMPARATORS:
        raise ValueError("Invalid qualification specifier: {}".format(qual))
    qual_id, comparator, value = parts

    if qual_id == "location":
        assert comparator in ["in", "not_in"], f"When using a location qualification, use 'in' or 'not_in'. Got {qual}."
        locations = value.split(",")

        value = []
        for location in locations:
            parts = location.split("-", 1)
            if len(parts) == 1:
                value.append({"Country": parts[0]})
            else:
                value.append({"Country": parts[0], "Subdivision": parts[1]})
        return {
            "QualificationTypeId": _QUAL_IDS.get(qual_id, qual_id),
            "Comparator": _COMPARATORS[comparator],
            "LocaleValues": value,
        }
    else:
        return {
            "QualificationTypeId": _QUAL_IDS.get(qual_id, qual_id),
            "Comparator": _COMPARATORS[comparator],
            "IntegerValues": [int(value)],
        }
# endregion


# region: hit management
def setup_hit_type(conn, props: dict) -> str:
    response = conn.create_hit_type(
        AutoApprovalDelayInSeconds=props.get('ApprovalTime', 2 * 24 * 60 * 60),  # 2 days.
        AssignmentDurationInSeconds=props.get('AssignmentTime', 20 * props['EstimatedTime']),
        # We don't want to have this be too long.
        Reward=f"{props['Reward']:0.2f}",
        Title=props['Title'],
        Keywords=props['Keywords'],
        Description=props['Description'],
        QualificationRequirements=[prepare_qualification(qual) for qual in props['Qualifications']],
    )
    return response['HITTypeId']


def launch_hit(conn, hit_type_id: str, config: dict, html: str) -> dict:
    """
    Launch a HIT using the configuration in `config` with contents `html`.

    Args:
        conn: A MTurk client connection
        hit_type_id: The HITTypeId returned by `setup_hit_type`. This ID
                     defines all the shared properties of tasks in this
                     HIT like their reward, and approval policy.
        config: The task configuration from ``task.yaml``.
        html: The HTML that will be rendered as part of this task.

    Returns:
        A dictionary describing the HIT.
    """
    response = conn.create_hit_with_hit_type(
        HITTypeId=hit_type_id,
        MaxAssignments=config['MaxAssignments'],
        LifetimeInSeconds=config.get("LifetimeInSeconds", 60 * 60 * 24 * 10),  # 10 days is default.
        Question=HTMLQuestion(html, config.get("FrameHeight", 1000)).get_as_xml(),
    )

    return {
        "HITId": response["HIT"]["HITId"],
        "HITTypeId": response["HIT"]["HITTypeId"],
        "CreationTime": "{:%Y%m%d %H:%M}".format(response["HIT"]["CreationTime"]),
        "HITStatus": response["HIT"]["HITStatus"],
        "MaxAssignments": response["HIT"]["MaxAssignments"],
        "NumberOfAssignmentsPending": response["HIT"]["NumberOfAssignmentsPending"],
        "NumberOfAssignmentsAvailable": response["HIT"]["NumberOfAssignmentsAvailable"],
        "NumberOfAssignmentsCompleted": response["HIT"]["NumberOfAssignmentsCompleted"],
        "NumberOfAssignmentsApproved": 0,
        "NumberOfAssignmentsRejected": 0,
    }


def redo_hit(conn, hit_id: str, num_redos: int = 1):
    """
    Creates additionall assignments to redo a hit.
    """
    conn.create_additional_assignments_for_hit(
        HITId=hit_id,
        NumberOfAdditionalAssignments=num_redos,
    )


def stop_hit(conn, hit: dict) -> dict:
    """
    Prevents any further action for this HIT.
    """
    conn.update_expiration_for_hit(
        HITId=hit['HITId'],
        ExpireAt=datetime.now(),
    )
    hit.update(get_hit(conn, hit['HITId']))
    return hit


def delete_hit(conn, hit: dict) -> dict:
    """
    Deletes a HIT (permanently).

    :param conn:
    :param hit:
    :return:
    """
    # First stop the hit.
    hit = stop_hit(conn, hit)

    # Then approve all pending assignments.
    if hit['NumberOfAssignmentsPending'] > 0:
        response = conn.list_assignments_for_hit(
            HITId=hit['HITId'],
            AssignmentStatuses=['Submitted'],
        )
        for assn in response['Assignments']:
            logger.warning("Approving assignment %s for HIT %s to delete this HIT", assn["AssignmentId"],
                           hit["HITId"])
            conn.approve_assignment(AssignmentId=assn['AssignmentId'])
            hit['NumberOfAssignmentsApproved'] += 1

        hit.update(get_hit(conn, hit['HITId']))
        assert hit['NumberOfAssignmentsPending'] == 0, (f"{hit['HITId']} still has "
                "{hit['NumberOfAssignmentsPending']} pending assignments")

    conn.delete_hit(HITId=hit['HITId'])
    hit['DeletedAt'] = DATETIME_FORMAT.format(datetime.now())
    return hit
# endregion


# region: hit+assn retrieval
def get_hit(conn, hit_id: str) -> dict:
    """
    Get a description corresponding to a HIT.
    This is most useful when updating statistics regarding a HIT.
    """
    response = conn.get_hit(HITId=hit_id)
    return {
        "HITId": response["HIT"]["HITId"],
        "HITTypeId": response["HIT"]["HITTypeId"],
        "CreationTime": DATETIME_FORMAT.format(response["HIT"]["CreationTime"]),
        "HITStatus": response["HIT"]["HITStatus"],
        "MaxAssignments": response["HIT"]["MaxAssignments"],
        "NumberOfAssignmentsPending": response["HIT"]["NumberOfAssignmentsPending"],
        "NumberOfAssignmentsAvailable": response["HIT"]["NumberOfAssignmentsAvailable"],
        "NumberOfAssignmentsCompleted": response["HIT"]["NumberOfAssignmentsCompleted"],
    }


def _extract_answer(answer: str) -> dict:
    """
    Extract an answer from an AMT response
    :param answer: An XML answer string with schema
        http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd
    :return: a dict containing key-value pairs.
    """
    ret = {}

    xmlns = '{http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd}'
    obj = ElementTree.fromstring(answer)
    for answer in obj.findall(f"{xmlns}Answer"):
        question = answer.find(f'{xmlns}QuestionIdentifier')

        if answer.find(f'{xmlns}FreeText') is not None:
            answer_value = answer.find(f'{xmlns}FreeText').text
        else:
            answer_value = None

        if answer_value and question.text.endswith("_as_json"):
            answer_value = json.loads(answer_value)
        ret[question.text] = answer_value
    return ret


def _parse_assn(assn: dict) -> dict:
    """
    Parses a raw assignment response from AMT.
    The answer is always in the 'Answer' field
    All other meta information is _Meta
    :param assn:
    :return:
    """
    return {
        "Answer": _extract_answer(assn['Answer']),
        "_Meta": {
            'AssignmentId': assn['AssignmentId'],
            'HITId': assn['HITId'],
            'WorkerId': assn['WorkerId'],
            'AssignmentStatus': assn['AssignmentStatus'],
            'SubmitTime': DATETIME_FORMAT.format(assn['SubmitTime']),
            'AcceptTime': DATETIME_FORMAT.format(assn['AcceptTime']),
            'WorkerTime': (assn['SubmitTime'] - assn['AcceptTime']).seconds,
        }
    }


def get_assn(conn, assn_id: str) -> dict:
    """
    Returns a parsed version of an assignment.
    :param conn:
    :param assn_id:
    :return: a dictionary with two keys, _Meta (for status, etc.) and Answer (for the actual data)
    """
    resp = conn.get_assignment(AssignmentId=assn_id)
    return _parse_assn(resp['Assignment'])


def sync_assn(conn, response: dict) -> dict:
    """
    Update an assignment response.
    :param conn:
    :param response:
    :return:
    """
    resp = conn.get_assignment(AssignmentId=response["_Meta"]['AssignmentId'])
    assn = resp['Assignment']

    response.update(_parse_assn(assn))
    return response


def sync_hit(conn, hit: dict, output: List[dict], force: bool = False) -> bool:
    """
    Update progress on this HIT. Are there any outputs pending? If so,
    walk through them to see if they need to be updated.

    @returns if we updated the HIT or Output.
    """
    seen_assns = {o["_Meta"]["AssignmentId"]: o for o in output if o["_Meta"]["AssignmentId"] is not None}

    # These are assignments that have been completed on paper, but we haven't yet synced them.
    unsynced_assns = hit['MaxAssignments'] - (
            hit['NumberOfAssignmentsAvailable'] + hit['NumberOfAssignmentsPending'] + len(seen_assns))

    # something could have changed.
    if force or hit['NumberOfAssignmentsAvailable'] > 0 or hit['NumberOfAssignmentsPending'] > 0 or unsynced_assns > 0:
        hit.update(get_hit(conn, hit['HITId']))

        # Updating this in case something changed above.
        unsynced_assns = hit['MaxAssignments'] - (
                hit['NumberOfAssignmentsAvailable'] + hit['NumberOfAssignmentsPending'] + len(seen_assns))

    # Check if local copy matches remote.
    if force or unsynced_assns > 0:
        # Get all responses.
        response = conn.list_assignments_for_hit(HITId=hit['HITId'])

        for assn in response['Assignments']:
            if assn['AssignmentId'] in seen_assns:
                seen_assns[assn['AssignmentId']].update(_parse_assn(assn))
            else:
                seen_assns[assn['AssignmentId']] = _parse_assn(assn)
                output.append(seen_assns[assn['AssignmentId']])
        return True
    else:
        # Nothing was sync-ed
        return False
# endregion


# region: payment
def approve_assignment(conn, output: dict, override_rejection: bool = True):
    """
    Approves the assignment in @output.
    At the end of this routine, output[_Meta][AssignmentStatus] will have been set to true.
    :param conn: MTurk Connection
    :param output: Assignment
    :param override_rejection: If true, we will override a rejection
    :return: None
    """
    meta = output["_Meta"]
    try:
        conn.approve_assignment(
            AssignmentId=meta['AssignmentId'],
            OverrideRejection=override_rejection)
        meta['AssignmentStatus'] = 'Approved'
    except ClientError as e:
        sync_assn(conn, output)
        meta = output["_Meta"]
        if meta['AssignmentStatus'] == 'Approved':
            logger.info("Assignment was already approved")
        else:
            raise e


def reject_assignment(conn, output: dict, feedback: str):
    """
    Rejects the assignment in @meta.
    At the end of this routine, meta[AssignmentStatus] will have been set to Rejected.
    :param conn: MTurk Connection
    :param output: Assignment response
    :param feedback: Message to be sent to the worker.
    :return: None
    """
    meta = output["_Meta"]
    try:
        conn.reject_assignment(
            AssignmentId=meta['AssignmentId'],
            RequesterFeedback=feedback)
        meta['AssignmentStatus'] = 'Rejected'
    except ClientError as e:
        sync_assn(conn, output)
        meta = output["_Meta"]
        if meta['AssignmentStatus'] == 'Approved':
            logger.error("Assignment was earlier approved and can't be rejected now")
        elif meta['AssignmentStatus'] == 'Rejected':
            logger.info("Assignment was already rejected")
        else:
            raise e


def pay_bonus(conn, worker_id: str, assn_id: str, amount: float = .5, reason: str = ''):
    conn.send_bonus(
        AssignmentId=assn_id,
        WorkerId=worker_id,
        BonusAmount=f"{amount:0.2f}",
        Reason=reason,
    )


def message_worker(conn, subject: str, message_text: str, worker_id: str):
    """
    Sends a message to the worker.
    """
    conn.notify_workers(
        Subject=subject,
        MessageText=message_text,
        WorkerIds=[worker_id],
    )
# endregion
