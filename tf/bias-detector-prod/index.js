const {spawn} = require('child_process');

let cors = require('cors')

let normalizePort = require('normalize-port')

let express = require('express')

let app = express()

app.use(cors())

var port = normalizePort(process.env.PORT || '4000');

app.get('/bias-detector', (req, res) => {
    var dataToSend;

    const args = [req.query.text]
    console.log("Before unshift: "+args)

    // const args = ["Google is a big company"]

    args.unshift("main.py")
    // args.unshift("python3")

    console.log(args)
  
    // const python = spawn("/Users/nikhilkulkarni/.pyenv/shims/python3.10", 'python3', args);

    // const python = spawn("/Users/nikhilkulkarni/.pyenv/shims/python3.10", args);
    // const python = spawn('python', args)

    const python = spawn('python3', args)

    python.stdout.on('data', function (data) {
        console.log('Pipe data from python script ...');
        dataToSend = data.toString();
    });
    
    python.on('close', (code) => {
    console.log(`child process close all stdio with code ${code}`);
    console.log(dataToSend)
    res.send(dataToSend)
    })
})

app.get('*', (req, res) => {
  res.send('<h1>Nikhil Kulkarni has created this server</h1>')
})

app.listen(port, () => {
    console.log('Node server is running @ http://localhost:4000')
})
