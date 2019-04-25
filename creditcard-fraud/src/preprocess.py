import time
import datetime


# Convert Time string to float value
def string_to_timestamp(date_string):  # convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


def __create_sets(amount):
    return [set() for _ in range(amount)]


def __create_dict(set_vals):
    result = {}
    for item in list(set_vals):
        result[item] = list(set_vals).index(item)

    return result

def cat_to_number(input_path, output_path):
    """
    Read the file of input_path and transform the data such that each categorical feature is mapped to a number.
    Note: this is just a improved version of the provided TA code
    :param input_path: path to original data
    :param output_path: path to output data
    :return:
    """
    data = []

    # Define sets to keep track of the encountered values (python sets can not contain duplicates)
    (issuercountry_set, txvariantcode_set, currencycode_set, shoppercountry_set, interaction_set,
     verification_set, accountcode_set, mail_id_set, ip_id_set, card_id_set) = __create_sets(10)

    # Read input file
    with open(input_path, 'r') as infile:
        # skip header
        infile.readline()

        for line_ah in infile:
            # Split the row into the different features
            line_data = line_ah.strip().split(',')

            # Remove rows with 'refused' label, since it's uncertain about fraud
            if line_data[9] == 'Refused':
                continue
            if 'na' in str(line_data[14]).lower() \
                    or 'na' in str(line_data[4].lower()):
                continue

            # Read individual features
            txid = line_data[0]
            bookingdate = string_to_timestamp(line_data[1])  # date reported fraud
            issuercountry = line_data[2]  # country code
            issuercountry_set.add(issuercountry)
            txvariantcode = line_data[3]  # type of card: visa/master
            txvariantcode_set.add(txvariantcode)
            issuer_id = float(line_data[4])  # bin card issuer identifier
            amount = float(line_data[5])  # transaction amount in minor units
            currencycode = line_data[6]
            currencycode_set.add(currencycode)
            shoppercountry = line_data[7]  # country code
            shoppercountry_set.add(shoppercountry)
            interaction = line_data[8]  # online transaction or subscription
            interaction_set.add(interaction)
            if line_data[9] == 'Chargeback':
                label = 1  # label fraud
            else:
                label = 0  # label save
            verification = line_data[10]  # shopper provide CVC code or not
            verification_set.add(verification)
            cvcresponse = int(line_data[11])  # 0 = Unknown, 1=Match, 2=No Match, 3-6=Not checked
            if cvcresponse > 2:
                cvcresponse = 3

            date_info = datetime.datetime.strptime(line_data[12], '%Y-%m-%d %H:%M:%S')
            year_info = date_info.year
            month_info = date_info.month
            day_info = date_info.day
            creationdate = str(year_info) + '-' + str(month_info) + '-' + str(day_info)  # Date of transaction
            creationdate_stamp = string_to_timestamp(line_data[12])  # Date of transaction-time stamp
            accountcode = line_data[13]  # merchant’s webshop
            accountcode_set.add(accountcode)
            mail_id = int(float(line_data[14].replace('email', '')))  # mail
            mail_id_set.add(mail_id)
            ip_id = int(float(line_data[15].replace('ip', '')))  # ip
            ip_id_set.add(ip_id)
            card_id = int(float(line_data[16].replace('card', '')))  # card
            card_id_set.add(card_id)

            # Store each row in the same order, but with the label at the end
            data.append([txid, bookingdate, issuercountry, txvariantcode, issuer_id, amount, currencycode,
                         shoppercountry, interaction, verification, cvcresponse, creationdate, creationdate_stamp,
                         accountcode, mail_id, ip_id, card_id, label])

    # Print some statistics of the file
    print("Total transcations: {}".format(len(data)))
    print("Total unique cards (also printed in TA script): {}".format(len(list(card_id_set))))

    # Map each categorical feature to a number
    issuercountry_dict = __create_dict(issuercountry_set)
    txvariantcode_dict = __create_dict(txvariantcode_set)
    currencycode_dict = __create_dict(currencycode_set)
    shoppercountry_dict = __create_dict(shoppercountry_set)
    interaction_dict = __create_dict(interaction_set)
    verification_dict = __create_dict(verification_set)
    accountcode_dict = __create_dict(accountcode_set)

    # TODO: these dicts contain to many elements?? function wont finish if included
    # mail_id_dict = __create_dict(mail_id_set)
    # ip_id_dict = __create_dict(ip_id_set)
    # card_id_dict = __create_dict(card_id_set)

    for item in data:
        item[2] = issuercountry_dict[item[2]]
        item[3] = txvariantcode_dict[item[3]]
        item[6] = currencycode_dict[item[6]]
        item[7] = shoppercountry_dict[item[7]]
        item[8] = interaction_dict[item[8]]
        item[9] = verification_dict[item[9]]
        item[13] = accountcode_dict[item[13]]
        # TODO: these dicts contain to many elements??
        # item[14] = mail_id_dict[item[14]]
        # item[15] = ip_id_dict[item[15]]
        # item[16] = card_id_dict[item[16]]

    # Now sort the same as original on txid
    data = sorted(data, key=lambda k: str(k))

    # Write transformed data
    print("Writing")
    with open(output_path, 'w+') as out:
        # Write header
        out.write('txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,' +
                  'currencycode,shoppercountrycode,shopperinteraction,cardverificationcodesupplied,cvcresponsecode' +
                  ',creationdate,createdate_stamp, accountcode, mail_id, ip_id, card_id, label')
        out.write('\n')

        for item in data:
            # Convert all elements of the item to string and join them
            row = ','.join(str(x) for x in item)
            out.write(row)
            # out.write(','.join(str(x) for x in item))
            out.write('\n')

    print("Done writing -- output file: {}".format(output_path))

    return True
