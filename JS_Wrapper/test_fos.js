var tools = require( "./fos.js" )

var fos = new tools.FOS();

var vectorized_X = new tools.VectorDouble();

var n = 20;
var p = 50;

for ( i = 0; i < n * p ; i++ ) {
    vectorized_X.push_back( Math.random() );
}

var Y = new tools.VectorDouble();

for ( i = 0; i < n * 1 ; i++ ) {
    Y.push_back( Math.random() );
}

var start = +(new Date);
fos.Run( vectorized_X, Y, "cd" )
var end = +(new Date);
var difference = end - start;

console.log( "Approximate execution time (ms): ", difference );

var coefs = fos.ReturnCoefficients();
var intercept = fos.ReturnIntercept();

console.log( "Intercept: ", intercept );
console.log( "Beta Vector:" );

for ( i = 0; i < coefs.size() ; i++ ) {
    console.log( coefs.get(i) );
}
