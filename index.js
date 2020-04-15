tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node')
fs = require('fs');

/**
 * Author : Colin Campbell
 * Note : Only a piece of test data was used; I don't want to destroy my machine
 */
let getData = function () {
    return new Promise((resolve, reject) => {
        fs.readFile('data/train.csv', (err, data) => {
            if (err) throw err;
            resolve(data.toString('utf8'))
        })
    })
}

getData().then((data) => {

    let rows = data.split('\n'); // split all rows
    rows.shift(); // remove the first row, ie the heading
    rows.pop() // there is an undefined row for the last row
    let feature = rows.map((val) => {
        let fields = val.split(',');
        fields.shift();
        //console.log(fields)
        let gender = fields[0] === 'Male' ? 1 : 0;
        let married = fields[1] === 'Yes' ? 1 : 0;
        let graduate = fields[3] === 'Graduate' ? 1 : 0;
        let selfEmployed = fields[4] === 'Yes' ? 1 : 0;
        let applicantIncome = parseInt(fields[5] || 0)
        let coApplicantIncome = parseInt(fields[6] || 0)
        let loanAmt = parseInt(fields[7] || 0)
        let loanTerm = parseInt(fields[8] || 0)
        let creditHistory = parseInt(fields[9] || 0)

        //let propertyArea = fields[10];

        return [gender, married, graduate, selfEmployed, applicantIncome,
            coApplicantIncome, loanAmt, loanTerm, creditHistory]
    })

    let label = rows.map((val) => {
        let fields = val.split(',');
        fields.shift();
        //console.log(fields)
        let loanStatus = fields[11];
        loanStatus = loanStatus.replace(/\s/g, '');

        let active;
        let notActive;
        if (loanStatus === 'Y') {
            active = 1
            notActive = 0
        } else {
            active = 0
            notActive = 1
        }

        return [active, notActive] // label data, placed in a 1 x 2 matrix so it can use classification instead of linear regression
    })

    console.log(feature[100])
    xs = tf.tensor2d(feature);
    ys = tf.tensor2d(label)

    let model = tf.sequential();

    // add layers
    model.add(tf.layers.dense({
        inputShape: [9],
        units: 10,
        activation: 'sigmoid' // sigmoid function returns the weights between a value of 0 & 1
    }))

    model.add(tf.layers.dense({
        inputShape: [10],
        units: 5,
        activation: 'sigmoid'
    }))

    model.add(tf.layers.dense({
        inputShape: [5],
        units: 2,
        activation: 'softmax' // softmac returns a probability
    }))

    model.compile({
        loss: 'meanSquaredError',
        optimizer: tf.train.adam(.06)
    })

    model.fit(xs, ys,
        {
            epochs: 400
        }).then((history) => {
            // sample feature is from the input
            model.predict(tf.tensor2d([
                [
                    1, 1, 0, 0,
                    4288, 3263, 133, 180,
                    1
                ]
            ])).print()
        })
})
